import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet101, ResNet101_Weights, ResNet
from torchvision.transforms.functional import affine, InterpolationMode
from torchvision.models.feature_extraction import create_feature_extractor


class PDN(torch.nn.Module):
    def __init__(self, backbone: ResNet, num_landmarks: int = 8,
                 num_classes: int = 200, num_attrs: int = 312,
                 landmark_dropout: float = 0.3) -> None:
        super().__init__()
        self.dim = backbone.layer4[0].conv1.in_channels + backbone.fc.in_features
        return_nodes = {
            'layer3.22.relu_2': 'layer3',
            'layer4.2.relu_2': 'layer4',
        }
        self.backbone = create_feature_extractor(backbone, return_nodes=return_nodes)
        self.num_landmarks = num_landmarks

        # New part of the model
        self.softmax = torch.nn.Softmax2d()
        self.fc_landmarks = torch.nn.Conv2d(1024 + 2048, num_landmarks + 1, 1, bias=False)
        self.modulation = torch.nn.Parameter(torch.ones((1,1024 + 2048,num_landmarks + 1)))
        self.dropout = torch.nn.Dropout(landmark_dropout)
        self.dropout_full_landmarks = torch.nn.Dropout1d(landmark_dropout)

        self.fc_attrs = torch.nn.Linear(1024 + 2048, num_attrs)
        self.fc_classes = torch.nn.Linear(num_attrs, num_classes, bias=False)
    
    def forward(self, x: torch.Tensor):
        # Pretrained ResNet part of the model
        features = self.backbone(x)
        l3, l4 = features['layer3'], features['layer4']
        b, c, h, w = l3.shape

        x = F.interpolate(l4, size=(h, w), mode='bilinear')  # shape: [b, 2048, h, w], e.g. h=w=14
        x = torch.cat([x, l3], dim=1)

        # Compute per landmark attention maps
        # (b - a)^2 = b^2 - 2ab + a^2, b = feature maps resnet, a = convolution kernel
        ab = self.fc_landmarks(x)
        b_sq = x.pow(2).sum(1, keepdim=True).expand(-1, self.num_landmarks + 1, -1, -1)
        a_sq = self.fc_landmarks.weight.sum(1).unsqueeze(0).expand(b, -1, h, w)
        maps = b_sq - 2 * ab + a_sq
        maps = self.softmax(-maps)

        # Use maps to get weighted average features per landmark
        feature_tensor = x
        # shape: [b,1,k,h,w],[b,c,1,h,w] -> [b,c,k]
        all_features = (maps.unsqueeze(1) * feature_tensor.unsqueeze(2)).mean((-1, -2))

        # Classification based on the landmarks
        all_features_modulated = all_features * self.modulation
        # all_features_modulated = self.dropout_full_landmarks(all_features_modulated.permute(0, 2, 1)).permute(0, 2, 1)
        # scores = self.fc_class_landmarks(all_features_modulated.permute(0, 2, 1)).permute(0, 2, 1)

        all_features_modulated = self.dropout_full_landmarks(all_features_modulated.permute(0, 2, 1))  # shape: [b,k,c]
        mean_features = torch.mean(all_features_modulated[:, :-1, :], dim=1)
        attr_scores = self.fc_attrs(mean_features)
        class_scores = self.fc_classes(attr_scores)

        return {
            'attr_scores': attr_scores,
            'class_scores': class_scores,
            'attn_maps': maps,
            'part_features': all_features
        }


def l_concentration(attn_maps: torch.Tensor) -> torch.Tensor:
    b, k, h, w = attn_maps.shape
    grid_x, grid_y = torch.meshgrid(torch.arange(w), torch.arange(h), indexing='xy')

    grid_x = grid_x[None, None, ...].to(attn_maps.device)
    grid_y = grid_y[None, None, ...].to(attn_maps.device)

    attn_map_sums = attn_maps.sum((-1, -2)).detach()
    cx = torch.sum(grid_x * attn_maps, dim=(-1, -2)) / attn_map_sums
    cy = torch.sum(grid_y * attn_maps, dim=(-1, -2)) / attn_map_sums

    spatial_var_x = ((cx[..., None, None] - grid_x) / w) ** 2
    spatial_var_y = ((cy[..., None, None] - grid_y) / h) ** 2
    spatial_var_weighted = (spatial_var_x + spatial_var_y) * attn_maps
    return spatial_var_weighted[:, 0:-1, :, :].mean()


def l_presence(attn_maps: torch.Tensor):
    maps_pooled = F.avg_pool2d(attn_maps[:, :, 2:-2, 2:-2], 3, stride=1)
    mean_presence = torch.amax(maps_pooled, dim=(0, 2, 3)).mean()
    return 1 - mean_presence


def l_orthogonality(num_parts: int, features: torch.Tensor) -> torch.Tensor:
    features = F.normalize(features, dim=1)
    similarities = features.permute(0, 2, 1) @ features
    similarities = similarities - torch.eye(num_parts + 1).to(features.device)
    return torch.mean(torch.square(similarities))


def rigid_transform(img: torch.Tensor, angle: int, translate: list[int], scale: float, invert: bool=False):
    shear = 0
    bilinear = InterpolationMode.BILINEAR
    if not invert:
        img = affine(img, angle, translate, scale, shear, interpolation=bilinear)
    else:
        translate = [-t for t in translate]
        img = affine(img, 0, translate, 1, shear)
        img = affine(img, -angle, [0, 0], 1/scale, shear)
    return img


def equiv_loss(X: torch.Tensor, maps: torch.Tensor, net: torch.nn.Module, device: torch.device, num_parts: int) \
        -> torch.Tensor:
    # Forward pass
    angle = np.random.rand() * 180 - 90
    translate = list(np.int32(np.floor(np.random.rand(2) * 100 - 50)))
    scale = np.random.rand() * 0.6 + 0.8
    transf_img = rigid_transform(X, angle, translate, scale, invert=False)
    outputs = net(transf_img.to(device))
    equiv_map = outputs['attn_maps']

    # Compare to original attention map, and penalise high difference
    translate = [(t * maps.shape[-1] / X.shape[-1]) for t in translate]
    rot_back = rigid_transform(equiv_map, angle, translate, scale, invert=True)
    num_elements_per_map = maps.shape[-2] * maps.shape[-1]
    orig_attmap_vector = torch.reshape(maps[:, :-1, :, :], (-1, num_parts, num_elements_per_map))
    transf_attmap_vector = torch.reshape(rot_back[:, 0:-1, :, :], (-1, num_parts, num_elements_per_map))
    cos_sim_equiv = F.cosine_similarity(orig_attmap_vector, transf_attmap_vector, -1)
    loss_equiv = 1 - torch.mean(cos_sim_equiv)
    return loss_equiv


class PDNLoss(nn.Module):
    def __init__(self, loss_coef_dict: dict[str, float], num_parts: int) -> None:
        super().__init__()
        self.loss_coef_dict = {k.lower(): v for k, v in loss_coef_dict.items()}
        self.l_cls = nn.CrossEntropyLoss()
        self.l_attr = nn.MSELoss()
        self.num_parts = num_parts
    
    def forward(self, model_outputs: dict[str, torch.Tensor], batch_inputs: dict[str, torch.Tensor], net: nn.Module):
        preds, attn_maps, part_features, attr_preds = (
            model_outputs['class_scores'],
            model_outputs['attn_maps'],
            model_outputs['part_features'],
            model_outputs['attr_scores']
        )
        images, labels = batch_inputs['pixel_values'], batch_inputs['class_ids']
        attr_scores = batch_inputs['attr_scores']
        device = labels.device
        loss_dict = {
            'l_cls': self.loss_coef_dict['l_cls'] * self.l_cls(preds, labels),
            'l_attr': self.loss_coef_dict['l_attr'] * self.l_attr(attr_preds, attr_scores),
            'l_conc': self.loss_coef_dict['l_conc'] * l_concentration(attn_maps),
            'l_pres': self.loss_coef_dict['l_pres'] * l_presence(attn_maps),
            'l_orth': self.loss_coef_dict['l_orth'] * l_orthogonality(self.num_parts, part_features),
            'l_equiv': self.loss_coef_dict['l_equiv'] * equiv_loss(images, attn_maps, net, device=device,
                                                                   num_parts=self.num_parts)
        }
        l_total = sum(loss_dict.values())
        loss_dict['l_total'] = l_total
        # l_cls = self.loss_coef_dict['l_cls'] * self.l_cls(model_outputs['class_scores'],
        #                                                    batch_inputs['class_ids']).mean()
        #
        # l_conc = self.loss_coef_dict['l_conc'] * conc_loss(loc_x, loc_y, grid_x, grid_y,
        #                                                     model_outputs['attn_maps'])
        #
        # l_pres = self.loss_coef_dict['l_pres'] * pres_loss(model_outputs['attn_maps'])
        #
        # l_orth = self.loss_coef_dict['l_orth'] * orth_loss(self.num_parts, model_outputs['part_features'],
        #                                                     model_outputs['part_features'].device)
        #
        # l_equiv = self.loss_coef_dict['l_equiv'] * equiv_loss(batch_inputs['pixel_values'],
        #                                                        model_outputs['attn_maps'],
        #                                                        net, device=model_outputs['attn_maps'].device,
        #                                                        num_parts=self.num_parts)
        # l_total = l_cls + l_conc + l_pres + l_orth + l_equiv
        # loss_dict = {'l_cls': l_cls, 'l_conc': l_conc,
        #              'l_pres': l_pres, 'l_orth': l_orth, 'l_equiv': l_equiv, 'l_total': l_total}
        return loss_dict


def load_pdn(loss_coef_dict: dict[str, float], lr: float, num_parts: int = 8):
    backbone = resnet101(weights=ResNet101_Weights.DEFAULT)
    pdn_net = PDN(backbone, num_landmarks=num_parts, num_classes=200)
    pdn_loss = PDNLoss(loss_coef_dict, num_parts)

    high_lr_layers = ["modulation"]
    med_lr_layers = ["fc_classes", "fc_attrs"]

    param_dict = [{'params': [], 'lr': lr * 100},
                  {'params': [], 'lr': lr * 10},
                  {'params': [], 'lr': lr}]
    for name, p in pdn_net.named_parameters():
        layer_name = name.split('.')[0]
        if layer_name in high_lr_layers:
            param_dict[0]['params'].append(p)
        elif layer_name in med_lr_layers:
            param_dict[1]['params'].append(p)
        else:
            param_dict[2]['params'].append(p)
    optimizer = torch.optim.Adam(params=param_dict)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.5)
    return pdn_net, pdn_loss, optimizer, scheduler
