import timm
import torch
from torch import nn
from torch import optim
import torch.nn.functional as f
from torch.optim import lr_scheduler
from omegaconf.dictconfig import DictConfig
from torchvision.models import resnet101, ResNet101_Weights


class APN(nn.Module):
    def __init__(self, attr_class_map: torch.Tensor, k: int = 112,
                 attr_groups: torch.Tensor = None, dist: str = 'dot') -> None:
        super().__init__()
        self.k = k
        self.backbone = timm.create_model('resnet101',
                                          pretrained=True,
                                          zero_init_last=False)
        self.dim = self.backbone.fc.weight.shape[-1]
        self.prototypes = nn.Parameter(torch.randn(self.k, self.dim))
        self.global_fc = nn.Linear(self.dim, self.k)
        self.final_fc = nn.Linear(self.k, 200, bias=False)
        self.final_fc.weight = nn.Parameter(attr_class_map)
        self.register_buffer('attr_groups', attr_groups)  # shape: [k]

        assert dist in ['dot', 'l2']
        self.dist = dist
    
    def forward(self, x):
        features = self.backbone.forward_features(x)
        b, c, h, w = features.shape
        features = features.view(b, c, h*w).permute(0, 2, 1)  # shape: [b,h*w,c]
        prototypes_batch = self.prototypes.unsqueeze(0).expand(b, -1, -1)  # shape: [b,k,c]

        if self.dist == 'dot':
            attn_maps = f.conv2d(features, prototypes_batch)  # shape: [b,k,h,2]
        elif self.dist == 'l2':
            attn_maps = 1 / torch.cdist(features, prototypes_batch, p=2).reshape(b, self.k, h, w)  # shape: [b,k,h,w]
        else:
            raise NotImplementedError

        max_attn_scores = f.avg_pool2d(attn_maps, kernel_size=(h, w))  # shape: [b,k,1,1]
        attr_scores = max_attn_scores.squeeze()  # shape: [b, k]

        class_scores = self.final_fc(attr_scores)

        max_attn_coords = torch.nonzero(attn_maps == max_attn_scores)
        max_attn_coords = max_attn_coords[..., 2:]  # shape: [b*k,2]

        # shape: [b,k], [b,k], [b,k,h,w], [k,dim], [b*k,2]
        return class_scores, attr_scores, attn_maps, self.prototypes, max_attn_coords


def decorrelation_loss(prototypes: torch.Tensor, group_idxs: torch.Tensor):
    group_weight_norms = []
    for i in torch.unique(group_idxs):
        mask = group_idxs == i
        weight_norm = torch.sum(torch.linalg.norm(prototypes[mask, :], ord=2, dim=0))
        group_weight_norms.append(weight_norm)
    return sum(group_weight_norms)


# TODO make this useful
def compactness_loss(attn_maps: torch.Tensor, max_logit_coords: torch.Tensor):
    (b, k, h, w), device = attn_maps.shape, attn_maps.device
    attn_maps = attn_maps.reshape(b*k, h, w)
    grid_w, grid_h = torch.meshgrid(torch.arange(w), torch.arange(h), indexing='xy')
    grid_w, grid_h = grid_w.to(device), grid_h.to(device)

    all_losses = []
    for m, coords in zip(attn_maps, max_logit_coords):
        # Expand coords of max attention scores for each attn_map m to shape [2, h, w] and unbind
        grid_ch, grid_cw = coords[..., None, None].expand(h, w, -1).unbind(dim=0)
        # High punishment if coords at away from center of attention still have high attention scores
        m_losses = m * ((grid_h - grid_ch) ** 2 + (grid_w - grid_cw) ** 2)
        all_losses.append(torch.mean(m_losses))

    return sum(all_losses) / len(all_losses)


def load_apn():
    return NotImplemented


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
                         lr: float) -> tuple[nn.Module, nn.Module, optim.Optimizer, lr_scheduler.LRScheduler]:
    net = BackBone(name=name, num_classes=num_classes)
    loss_fn = BackboneFinetuneLoss()

    optimizer = optim.AdamW(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 5, 0.5)
    return net, loss_fn, optimizer, scheduler
