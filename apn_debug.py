import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def add_glasso(var, group):
    return var[group, :].pow(2).sum(dim=0).add(1e-8).sum().pow(1/2.)

def get_middle_graph(weight_cpt, model):
    middle_graph = None
    if weight_cpt > 0:
        # creat middle_graph to mask the L_CPT:
        kernel_size = model.kernel_size[model.extract[0]]
        raw_graph = torch.zeros((2 * kernel_size - 1, 2 * kernel_size - 1))
        for x in range(- kernel_size + 1, kernel_size):
            for y in range(- kernel_size + 1, kernel_size):
                raw_graph[x + (kernel_size - 1), y + (kernel_size - 1)] = x ** 2 + y ** 2
        middle_graph = torch.zeros((kernel_size ** 2, kernel_size, kernel_size))
        for x in range(kernel_size):
            for y in range(kernel_size):
                middle_graph[x * kernel_size + y, :, :] = \
                    raw_graph[kernel_size - 1 - x: 2 * kernel_size - 1 - x,
                    kernel_size - 1 - y: 2 * kernel_size - 1 - y]
        middle_graph = middle_graph.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return middle_graph


class resnet_proto_IoU(nn.Module):
    def __init__(self, opt):
        super(resnet_proto_IoU, self).__init__()
        resnet = models.resnet101()
        num_ftrs = resnet.fc.in_features
        num_fc_dic = {'cub':150, 'awa2': 40, 'sun': 645}

        if 'c' in opt.resnet_path:
            num_fc = num_fc_dic['cub']
        elif 'awa2' in opt.resnet_path:
            num_fc = num_fc_dic['awa2']
        elif 'sun' in opt.resnet_path:
            num_fc = num_fc_dic['sun']
        else:
            num_fc = 1000
        resnet.fc = nn.Linear(num_ftrs, num_fc)

        # 01 - load resnet to model1
        if opt.resnet_path is not None:
            state_dict = torch.load(opt.resnet_path, map_location='cpu')
            resnet.load_state_dict(state_dict)
            # print("resnet load state dict from {}".format(opt.resnet_path))

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fine_tune(True)

        # 02 - load cls weights
        # we left the entry for several layers, but here we only use layer4
        self.dim_dict = {'layer1': 56*56, 'layer2': 28*28, 'layer3': 14*14, 'layer4': 7*7, 'avg_pool': 1*1}
        self.channel_dict = {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048, 'avg_pool': 2048}
        self.kernel_size = {'layer1': 56, 'layer2': 28, 'layer3': 14, 'layer4': 7, 'avg_pool': 1}
        self.extract = ['layer4']  # 'layer1', 'layer2', 'layer3', 'layer4'
        self.epsilon = 1e-4

        self.softmax = nn.Softmax(dim=1)
        self.softmax2d = nn.Softmax2d()
        self.sigmoid = nn.Sigmoid()
        if opt.dataset == 'CUB':
            self.prototype_vectors = dict()
            for name in self.extract:
                prototype_shape = [312, self.channel_dict[name], 1, 1]
                self.prototype_vectors[name] = nn.Parameter(2e-4 * torch.rand(prototype_shape), requires_grad=True)
            self.prototype_vectors = nn.ParameterDict(self.prototype_vectors)
            self.ALE_vector = nn.Parameter(2e-4 * torch.rand([312, 2048, 1, 1]), requires_grad=True)
        elif opt.dataset == 'AWA1':
            exit(1)
        elif opt.dataset == 'AWA2':
            self.prototype_vectors = dict()
            for name in self.extract:
                prototype_shape = [85, self.channel_dict[name], 1, 1]
                self.prototype_vectors[name] = nn.Parameter(2e-4 * torch.rand(prototype_shape), requires_grad=True)
            self.prototype_vectors = nn.ParameterDict(self.prototype_vectors)
            self.ALE_vector = nn.Parameter(2e-4 * torch.rand([85, 2048, 1, 1]), requires_grad=True)
        elif opt.dataset == 'SUN':
            self.prototype_vectors = dict()
            for name in self.extract:
                prototype_shape = [102, self.channel_dict[name], 1, 1]
                self.prototype_vectors[name] = nn.Parameter(2e-4 * torch.rand(prototype_shape), requires_grad=True)
            self.prototype_vectors = nn.ParameterDict(self.prototype_vectors)
            self.ALE_vector = nn.Parameter(2e-4 * torch.rand([102, 2048, 1, 1]), requires_grad=True)
        self.avg_pool = opt.avg_pool

    def forward(self, x, attribute, return_map=False):
        """out: predict class, predict attributes, maps, out_feature"""
        # print('x.shape', x.shape)
        record_features = {}
        batch_size = x.size(0)
        x = self.resnet[0:5](x)  # layer 1
        record_features['layer1'] = x  # [64, 256, 56, 56]
        x = self.resnet[5](x)  # layer 2
        record_features['layer2'] = x  # [64, 512, 28, 28]
        x = self.resnet[6](x)  # layer 3
        record_features['layer3'] = x  # [64, 1024, 14, 14]
        x = self.resnet[7](x)  # layer 4
        record_features['layer4'] = x  # [64, 2048, 7, 7]

        attention = dict()
        pre_attri = dict()
        pre_class = dict()

        if self.avg_pool:
            pre_attri['final'] = F.avg_pool2d(F.conv2d(input=x, weight=self.ALE_vector), kernel_size=7).view(batch_size, -1)
        else:
            pre_attri['final'] = F.max_pool2d(F.conv2d(input=x, weight=self.ALE_vector), kernel_size=7).view(batch_size, -1)
        # print("pre_attri['final'].shape:", pre_attri['final'].shape)
        # print("attribute.shape:", attribute.shape)
        # exit()
        output_final = self.softmax(pre_attri['final'].mm(attribute))

        for name in self.extract:
            # print("hererererere:", record_features[name].shape)
            attention[name] = F.conv2d(input=record_features[name], weight=self.prototype_vectors[name])  # [64, 312, W, H]
            pre_attri[name] = F.max_pool2d(attention[name], kernel_size=self.kernel_size[name]).view(batch_size, -1)
            pre_class[name] = self.softmax(pre_attri[name].mm(attribute))
        return output_final, pre_attri, attention, pre_class

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

    def _l2_convolution(self, x, prototype_vector, one):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''
        x2 = x ** 2  # [64, C, W, H]
        x2_patch_sum = F.conv2d(input=x2, weight=one)

        p2 = prototype_vector ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=prototype_vector)
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast  [64, 312,  W, H]
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)  # [64, 312,  W, H]
        return distances


def Loss_fn(opt, loss_log, reg_weight, criterion, criterion_regre, model,
            output, pre_attri, attention, pre_class, label_a, label_v,
            realtrain, middle_graph, parts, group_dic, sub_group_dic):
    # for Layer_Regression:
    loss = 0
    if reg_weight['final']['xe'] > 0:
        loss_xe = reg_weight['final']['xe'] * criterion(output, label_v)
        loss_log['l_xe_final'] += loss_xe.item()
        loss = loss_xe

    if reg_weight['final']['attri'] > 0:
        loss_attri = reg_weight['final']['attri'] * criterion_regre(pre_attri['final'], label_a)
        loss_log['l_attri_final'] += loss_attri.item()
        loss += loss_attri

    # add regularization loss
    if opt.additional_loss:
        weight_final = model.ALE_vector.squeeze()
        for name in model.extract:  # "name" is layer4 currently
            if reg_weight[name]['l_xe'] > 0:
                layer_xe = reg_weight[name]['l_xe'] * criterion(pre_class[name], label_v)
                loss_log['l_xe_layer'] += layer_xe.item()
                loss += layer_xe

            if reg_weight[name]['attri'] > 0:
                loss_attri = reg_weight[name]['attri'] * criterion_regre(pre_attri[name], label_a)
                loss_log['l_attri_layer'] += loss_attri.item()
                loss += loss_attri

            weight_layer = model.prototype_vectors[name].squeeze()

            batch_size, attri_dim, map_dim, map_dim = attention[name].size()

            if opt.use_group:
                batch_size, attri_dim, map_dim, map_dim = attention[name].size()
                if reg_weight[name]['cpt'] > 0 and realtrain:
                    peak_id = torch.argmax(attention[name].view(batch_size * attri_dim, -1), dim=1)
                    peak_mask = middle_graph[peak_id, :, :].view(batch_size, attri_dim, map_dim, map_dim)
                    cpt_loss = reg_weight[name]['cpt'] * torch.sum(model.sigmoid(attention[name]) * peak_mask)
                    loss_log['l_cpt'] += cpt_loss.item()
                    loss += cpt_loss

                for part in parts[:7]:
                    group = group_dic[part]
                    # res_loss
                    if reg_weight[name]['regular'] > 0:
                        reg_loss = reg_weight[name]['regular'] * add_glasso(weight_layer, group)
                        loss_log['l_regular_layer'] += reg_loss.item()
                        loss += reg_loss

                    if reg_weight['final']['regular'] > 0:
                        reg_loss = reg_weight['final']['regular'] * add_glasso(weight_final, group)
                        loss_log['l_regular_final'] += reg_loss.item()
                        loss += reg_loss
            else:
                if reg_weight[name]['regular'] > 0:
                    reg_loss = reg_weight[name]['regular'] * weight_layer.norm(2)
                    loss_log['l_regular_layer'] += reg_loss.item()
                    loss += reg_loss
                    reg_loss = reg_weight['final']['regular'] * weight_final.norm(2)
                    loss_log['l_regular_final'] += reg_loss.item()
                    loss += reg_loss

            # if reg_weight[name]['cpt'] > 0 and realtrain:
            #     peak_id = torch.argmax(attention[name].view(batch_size * attri_dim, -1), dim=1)
            #     peak_mask = middle_graph[peak_id, :, :].view(batch_size, attri_dim, map_dim, map_dim)
            #     cpt_loss = reg_weight[name]['cpt'] * torch.sum(model.sigmoid(attention[name]) * peak_mask)
            #     loss_log['l_cpt'] += cpt_loss.item()
            #     loss += cpt_loss

            # for part in parts[:7]:
            #     group = group_dic[part]
            #     # res_loss
            #     if reg_weight[name]['regular'] > 0:
            #         reg_loss = reg_weight[name]['regular'] * add_glasso(weight_layer, group)
            #         loss_log['l_regular_layer'] += reg_loss.item()
            #         loss += reg_loss
            #
            #     if reg_weight['final']['regular'] > 0:
            #         reg_loss = reg_weight['final']['regular'] * add_glasso(weight_final, group)
            #         loss_log['l_regular_final'] += reg_loss.item()
            #         loss += reg_loss
    return loss
