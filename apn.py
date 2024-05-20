import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torchvision.models import resnet101, ResNet101_Weights, ResNet
# from torchvision.models.feature_extraction import create_feature_extractor


########################################
# Modified Attribute Prototype Network #
########################################

class APN(nn.Module):
    def __init__(
            self,
            backbone: nn.Module,
            class_embeddings: torch.Tensor,
            dist: str = 'dot'
        ) -> None:
        super().__init__()
        if isinstance(backbone, ResNet):
            self.backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
            self.dim = backbone.fc.in_features
        else:
            raise NotImplementedError
        self.num_classes, self.num_attrs = class_embeddings.shape

        self.register_buffer('class_embeddings', class_embeddings)
        self.attr_prototypes = nn.Parameter(torch.zeros(self.num_attrs, self.dim))

        assert dist in ['dot', 'l2']
        self.dist = dist

    def forward(self, batch_inputs: dict[str, torch.Tensor]):
        x = batch_inputs['pixel_values']

        features = self.backbone(x)  # type: torch.Tensor
        b, c, h, w = features.shape

        if self.dist == 'dot':
            attn_maps = F.conv2d(features, self.attr_prototypes[..., None, None])  # shape: [b,k,h,w]
        else:
            raise NotImplementedError

        max_attn_scores = F.max_pool2d(attn_maps, kernel_size=(h, w))  # shape: [b,k,1,1]
        attr_scores = max_attn_scores.squeeze()  # shape: [b, k]

        class_scores = attr_scores @ self.class_embeddings.T

        # shape: [b,num_classes], [b,k], [b,k,h,w]
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
        l_cls = self.loss_coef_dict['l_cls'] * self.l_cls(model_outputs['class_scores'], batch_inputs['class_ids'])
        l_reg = self.loss_coef_dict['l_reg'] * self.l_reg(model_outputs['attr_scores'], batch_inputs['attr_scores'])
        l_total = l_cls + l_reg
        loss_dict = {
            'l_cls': l_cls.item(),
            'l_reg': l_reg.item(),
            'l_total': l_total.item()
        }
        return loss_dict, l_total

    @staticmethod
    def l_cpt(attn_maps: torch.Tensor):
        # TODO Fix this loss. It prevents the model from learning.
        (b, k, h, w), device = attn_maps.shape, attn_maps.device
        grid_w, grid_h = torch.meshgrid(torch.arange(w), torch.arange(h), indexing='xy')
        grid_w, grid_h = grid_w.to(device), grid_h.to(device)

        # Compute coordinates of max attention scores
        max_attn_scores = F.max_pool2d(attn_maps, kernel_size=(h, w))  # shape: [b,k,1,1]
        max_attn_coords = torch.nonzero(attn_maps == max_attn_scores)  # shape: [b*k,4]
        max_attn_coords = max_attn_coords[..., 2:]  # shape: [b*k,2]

        attn_maps = F.sigmoid(attn_maps.reshape(b*k, h, w))  # range: [0,1]

        all_losses = []
        for m, coords in zip(attn_maps, max_attn_coords):
            # Expand coords of max attention scores for each attn_map m to shape [2,h,w] and unbind
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
        backbone_name: str,
        backbone_weights_path: str,
        class_embeddings: torch.Tensor,
        loss_coef_dict: dict[str, float],
        dist: str,
        lr: float,
        betas: tuple[float, float],
        step_size: int,
        gamma: float
    ) -> tuple[nn.Module, nn.Module, optim.Optimizer, lr_scheduler.LRScheduler]:
    if backbone_name == 'resnet101':
        num_classes, num_attrs = class_embeddings.shape
        backbone = resnet101(weights=ResNet101_Weights.DEFAULT)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
    else:
        raise NotImplementedError
    backbone_weights = torch.load(backbone_weights_path, map_location='cpu')
    backbone.load_state_dict(backbone_weights)
    apn_net = APN(backbone, class_embeddings, dist=dist)
    apn_loss = APNLoss(loss_coef_dict)

    optimizer = optim.AdamW(params=[apn_net.attr_prototypes], lr=lr, betas=betas)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return apn_net, apn_loss, optimizer, scheduler


########################
# Backbone fine-tuning #
########################

class BackBone(nn.Module):
    def __init__(self, name: str, num_classes: int):
        super().__init__()
        assert name in ['resnet101', 'ViT-L/16']
        if name == 'resnet101':
            self.backbone = resnet101(weights=ResNet101_Weights.DEFAULT)
        else:
            raise NotImplementedError
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
