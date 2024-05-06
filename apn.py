import timm
import torch
from torch import nn
import torch.nn.functional as F


class APN(nn.Module):
    def __init__(self, attr_class_map: torch.Tensor, k=112, attr_groups=None) -> None:
        super().__init__()
        self.k = k
        self.backbone = timm.create_model('resnet101',
                                          pretrained=True,
                                          zero_init_last=False)
        self.dim = self.backbone.fc.weight.shape[-1]
        self.prototypes = nn.Parameter(torch.randn(self.k, self.dim))
        self.global_fc = nn.Linear(self.dim, self.k)
        self.register_buffer('attr_class_map', attr_class_map)  # shape: [k, num_classes]
        self.register_buffer('attr_groups', attr_groups)  # shape: [k]
    
    def forward(self, x):
        features = self.backbone.forward_features(x)

        # Global branch
        feature_global = self.backbone.global_pool(features)
        global_logits = self.global_fc(feature_global) @ self.attr_class_map.T
        # Local branch
        attn_maps = torch.einsum('bchw,kc->bkhw', features, self.prototypes)
        max_local_logits = torch.amax(attn_maps, dim=(-1, -2), keepdim=True)
        local_logits = max_local_logits.squeeze()  # shape: [b, k]

        max_logits_coords = torch.nonzero(attn_maps == max_local_logits)
        max_logits_coords = max_logits_coords[..., 2:]  # shape: [b*k,2]

        # shape: [b, k], [b, k], [b, k, h, w], [k, dim], [b*k, 2]
        return global_logits, local_logits, attn_maps, self.prototypes, max_logits_coords


def decorrelation_loss(prototypes: torch.Tensor, group_idxs: torch.Tensor):
    group_weight_norms = []
    for i in torch.unique(group_idxs):
        mask = group_idxs == i
        weight_norm = torch.sum(torch.linalg.norm(prototypes[mask, :], ord=2, dim=0))
        group_weight_norms.append(weight_norm)
    return sum(group_weight_norms)


def compactness_loss(attn_maps: torch.Tensor, max_logit_coords: torch.Tensor):
    (b, k, h, w), device = attn_maps.shape, attn_maps.device
    attn_maps = attn_maps.reshape(b*k, h, w)
    grid_w, grid_h = torch.meshgrid(torch.arange(w), torch.arange(h), indexing='xy')
    grid_w, grid_h = grid_w.to(device), grid_h.to(device)

    all_losses = []
    for m, coords in zip(attn_maps, max_logit_coords):
        # Expand coord of max logit for attn_map m to shape [h, w, 2] and unbind
        grid_ch, grid_cw = coords[None, None, ...].expand(h, w, -1).unbind(dim=-1)
        m_losses = m * ((grid_h - grid_ch) ** 2 + (grid_w - grid_cw) ** 2)
        all_losses.append(torch.sum(m_losses))

    return sum(all_losses) 
