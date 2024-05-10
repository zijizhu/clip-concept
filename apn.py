import timm
import torch
from torch import nn
from torch import optim
import torch.nn.functional as f
from torch.optim import lr_scheduler


########################################
# Modified Attribute Prototype Network #
########################################

class APN(nn.Module):
    def __init__(self, num_classes: int, num_attrs: int, backbone_name: str,
                 backbone_weights: dict[str, torch.Tensor], dist: str = 'dot') -> None:
        super().__init__()
        self.k = num_attrs
        self.backbone = timm.create_model(backbone_name,
                                          pretrained=True,
                                          zero_init_last=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.dim = self.backbone.fc.weight.shape[-1]

        self.prototypes = nn.Parameter(torch.randn(self.k, self.dim))
        self.final_fc = nn.Linear(self.k, num_classes)

        self.backbone.load_state_dict(backbone_weights)

        assert dist in ['dot', 'l2']
        self.dist = dist
    
    def forward(self, batch_inputs):
        x = batch_inputs['pixel_values']

        features = self.backbone.forward_features(x)
        b, c, h, w = features.shape

        if self.dist == 'dot':
            attn_maps = f.conv2d(features, self.prototypes[..., None, None])  # shape: [b,k,h,w]
        elif self.dist == 'l2':
            features = features.view(b, c, h*w).permute(0, 2, 1)  # shape: [b,h*w,c]
            prototypes_batch = self.prototypes.unsqueeze(0).expand(b, -1, -1)  # shape: [b,k,c]
            attn_maps = 1 / torch.cdist(features, prototypes_batch, p=2).reshape(b, self.k, h, w)  # shape: [b,k,h,w]
        else:
            raise NotImplementedError

        max_attn_scores = f.max_pool2d(attn_maps, kernel_size=(h, w))  # shape: [b,k,1,1]
        attr_scores = max_attn_scores.squeeze()  # shape: [b, k]

        class_scores = self.final_fc(attr_scores)

        # shape: [b,k], [b,k], [b,k,h,w]
        return {
            'class_scores': class_scores,
            'attr_scores': attr_scores,
            'attn_maps': attn_maps
        }


class APNLoss(nn.Module):
    def __init__(self, loss_coef_dict: dict[str, float]):
        super().__init__()
        self.loss_coef_dict = {k.lower(): v for k, v in loss_coef_dict.items()}
        self.l_cls = nn.CrossEntropyLoss()
        self.l_reg = nn.MSELoss()

    def forward(self, model_outputs: dict[str, torch.Tensor], batch_inputs: dict[str, torch.Tensor]):
        loss_dict = {
            'l_cls': self.loss_coef_dict['l_cls'] * self.l_cls(model_outputs['class_scores'], batch_inputs['class_ids']),
            'l_reg': self.loss_coef_dict['l_reg'] * self.l_reg(model_outputs['attr_scores'], batch_inputs['attr_scores']),
            'l_cpt': self.loss_coef_dict['l_cpt'] * self.l_cpt(model_outputs['attn_maps'])
        }
        l_total = sum(loss for loss in loss_dict.values())
        loss_dict['l_total'] = l_total
        return loss_dict

    @staticmethod
    def l_cpt(attn_maps: torch.Tensor):
        (b, k, h, w), device = attn_maps.shape, attn_maps.device
        grid_w, grid_h = torch.meshgrid(torch.arange(w), torch.arange(h), indexing='xy')
        grid_w, grid_h = grid_w.to(device), grid_h.to(device)

        # Compute coordinates of max attention scores
        max_attn_scores = f.max_pool2d(attn_maps, kernel_size=(h, w))  # shape: [b,k,1,1]
        max_attn_coords = torch.nonzero(attn_maps == max_attn_scores)  # shape: [b*k, 4]
        max_attn_coords = max_attn_coords[..., 2:]  # shape: [b*k,2]

        attn_maps = attn_maps.reshape(b*k, h, w)

        all_losses = []
        for m, coords in zip(attn_maps, max_attn_coords):
            # Expand coords of max attention scores for each attn_map m to shape [2, h, w] and unbind
            grid_ch, grid_cw = coords[..., None, None].expand(-1, h, w).unbind(dim=0)
            # High punishment if coords at away from center of attention still have high attention scores
            m_losses = m * ((grid_h - grid_ch) ** 2 + (grid_w - grid_cw) ** 2)
            all_losses.append(torch.mean(m_losses))

        return sum(all_losses) / len(all_losses)

    @staticmethod
    def l_decorrelation(prototypes: torch.Tensor, group_idxs: torch.Tensor):
        group_weight_norms = []
        for i in torch.unique(group_idxs):
            mask = group_idxs == i
            weight_norm = torch.sum(torch.linalg.norm(prototypes[mask, :], ord=2, dim=0))
            group_weight_norms.append(weight_norm)
        return sum(group_weight_norms)


def load_apn(
        num_classes: int,
        num_attrs: int,
        dist: str, lr: float,
        backbone_name: str,
        backbone_weight_path: str,
        loss_coef_dict: dict[str, float],
        step_size: int,
        gamma: float
    ) -> tuple[nn.Module, nn.Module, optim.Optimizer, lr_scheduler.LRScheduler]:
    backbone_weights = torch.load(backbone_weight_path, map_location='cpu')
    apn_net = APN(num_classes=num_classes, num_attrs=num_attrs,
                  backbone_name=backbone_name, backbone_weights=backbone_weights, dist=dist)
    apn_loss = APNLoss(loss_coef_dict)

    optimizer = optim.AdamW(apn_net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return apn_net, apn_loss, optimizer, scheduler


########################
# Backbone fine-tuning #
########################

class BackBone(nn.Module):
    def __init__(self, name: str, num_classes: int):
        super().__init__()
        assert name in ['resnet101', 'ViT-L/16']
        # if name == 'resnet101':
        # self.backbone = resnet101(weights=ResNet101_Weights.DEFAULT)
        self.backbone = timm.create_model(name, pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, batch_inputs: dict[str, torch.Tensor]):
        class_scores = self.backbone(batch_inputs['pixel_values'])
        return {'class_scores': class_scores}

    def state_dict(self, *args, **kwargs):
        return self.backbone.state_dict()


class BackboneFinetuneLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, model_outputs: dict[str, torch.Tensor], batch_inputs: dict[str, torch.Tensor]):
        ce_loss = self.loss(model_outputs['class_scores'], batch_inputs['class_ids'])
        return {'l_total': ce_loss}


@torch.no_grad()
def compute_corrects(model_outputs: dict[str, torch.Tensor], batch_inputs: dict[str, torch.Tensor]):
    class_scores, class_ids = model_outputs['class_scores'], batch_inputs['class_ids']
    return torch.sum(torch.argmax(class_scores.data, dim=-1) == class_ids.data).item()


def load_backbone_for_ft(name: str,
                         num_classes: int,
                         lr: float,
                         step_size: int,
                         gamma: float) -> tuple[nn.Module, nn.Module, optim.Optimizer, lr_scheduler.LRScheduler]:
    net = BackBone(name=name, num_classes=num_classes)
    loss_fn = BackboneFinetuneLoss()

    optimizer = optim.AdamW(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
    return net, loss_fn, optimizer, scheduler
