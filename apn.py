import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision.models import ResNet, ResNet101_Weights, resnet101

########################################
# Modified Attribute Prototype Network #
########################################


class APN(nn.Module):
    def __init__(self, backbone: nn.Module, class_embeddings: torch.Tensor, dist: str = "dot") -> None:
        super().__init__()
        if isinstance(backbone, ResNet):
            self.backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
            self.dim = backbone.fc.in_features
        else:
            raise NotImplementedError
        self.num_classes, self.num_attrs = class_embeddings.shape

        self.register_buffer("class_embeddings", class_embeddings)
        # self = nn.Conv2d(self.dim, self.num_attrs, kernel_size=(1, 1), bias=False)
        self.prototypes = nn.Parameter(nn.init.normal_(torch.empty(self.num_attrs, self.dim), std=1e-2))

        assert dist in ["dot", "l2"]
        self.dist = dist

    def forward(self, batch: dict[str, torch.Tensor]):
        x = batch["pixel_values"]

        features = self.backbone(x)  # type: torch.Tensor
        b, c, h, w = features.shape

        if self.dist == "dot":
            attn_maps = F.conv2d(features, self.prototypes[..., None, None])  # shape: [b,k,h,w]
            # attn_maps = self.conv(features)
        else:
            raise NotImplementedError

        max_attn_scores = F.max_pool2d(attn_maps, kernel_size=(h, w))
        attr_scores = max_attn_scores.squeeze()  # shape: [b, k]

        class_scores = attr_scores @ self.class_embeddings.T  # type: torch.Tensor

        # shape: [b,num_classes], [b,k], [b,k,h,w]
        return {
            "class_scores": class_scores,
            "attr_scores": attr_scores,
            "attn_maps": attn_maps,
            "prototypes": self.prototypes
        }


class APNLoss(nn.Module):
    def __init__(self, group_ids: torch.Tensor, **kwargs):
        super().__init__()
        self.l_cls_coef = kwargs["l_cls"]  # type: int
        self.l_reg_coef = kwargs["l_reg"]  # type: int
        self.l_cpt_coef = kwargs["l_cpt"]  # type: int
        self.l_dec_coef = kwargs["l_dec"]  # type: int

        self.l_cls = nn.CrossEntropyLoss()
        self.l_reg = nn.MSELoss()
        
        self.group_ids = group_ids

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]):
        loss_dict = {
            "l_cls": self.l_cls_coef * self.l_cls(outputs["class_scores"], batch["class_ids"]),
            "l_reg": self.l_reg_coef * self.l_reg(outputs["attr_scores"], batch["attr_scores"]),
            "l_cpt": self.l_cpt_coef * self.l_cpt(outputs["attn_maps"]),
            "l_dec": self.l_dec_coef * self.l_dec(outputs["prototypes"], self.group_ids)
        }
        l_total = sum(loss_dict.values())
        return loss_dict, l_total

    @staticmethod
    def l_cpt(attn_maps: torch.Tensor):
        '''Loss function for compactness of attention maps'''
        device = attn_maps.device
        b, k, h, w = attn_maps.shape
        grid_w, grid_h = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")
        grid_w, grid_h = grid_w.to(device), grid_h.to(device)

        # Compute coordinates of max attention scores, shape: [b,k]
        _, max_attn_indices = F.max_pool2d(attn_maps.detach(), kernel_size=(h, w), return_indices=True)
        max_attn_indices = max_attn_indices.squeeze()
        max_attn_h, max_attn_w = torch.unravel_index(max_attn_indices, shape=(h, w))  # shape: [b,k], [b,k]
        max_attn_h = max_attn_h[..., None, None].expand(-1, -1, h, w)  # shape: [b,k,h,w]
        max_attn_w = max_attn_w[..., None, None].expand(-1, -1, h, w)  # shape: [b,k,h,w]

        attn_maps = F.sigmoid(attn_maps)  # shape: [b*k,h,w], range: [0,1]
        losses = attn_maps * ((grid_h - max_attn_h) ** 2 + (grid_w - max_attn_w) ** 2)  # shape: [b*k,h,w]

        return torch.mean(losses)

    @staticmethod
    def l_dec(prototypes: torch.Tensor, group_ids: torch.Tensor):
        '''Loss function for decorrelation of attribute groups'''
        all_group_losses = []
        for i in torch.unique(group_ids):
            mask = group_ids == i
            group_loss = prototypes[mask, :].pow(2).sum().sqrt()
            all_group_losses.append(group_loss)
        return sum(all_group_losses)


def load_apn(
    backbone_name: str,
    class_embeddings: torch.Tensor,
    attr_group_ids: torch.Tensor,
    loss_coef_dict: dict[str, float],
    dist: str,
    lr: float,
    betas: tuple[float, float],
    step_size: int,
    gamma: float,
) -> tuple[nn.Module, nn.Module, optim.Optimizer, lr_scheduler.LRScheduler]:
    if backbone_name == "resnet101":
        num_classes, num_attrs = class_embeddings.shape
        backbone = resnet101(weights=ResNet101_Weights.DEFAULT)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
    else:
        raise NotImplementedError
    apn_net = APN(backbone, class_embeddings, dist=dist)
    apn_loss = APNLoss(attr_group_ids, **{k.lower(): v for k, v in loss_coef_dict.items()})

    optimizer = optim.AdamW(
        params=[
            {"params": apn_net.backbone.parameters(), "lr": lr * 0.1},
            {"params": apn_net.prototypes},
        ],
        lr=lr,
        betas=betas,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return apn_net, apn_loss, optimizer, scheduler


########################
# Backbone fine-tuning #
########################


class BackBone(nn.Module):
    def __init__(self, name: str, num_classes: int):
        super().__init__()
        assert name in ["resnet101", "ViT-L/16"]
        if name == "resnet101":
            self.backbone = resnet101(weights=ResNet101_Weights.DEFAULT)
        else:
            raise NotImplementedError
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, batch: dict[str, torch.Tensor]):
        class_scores = self.backbone(batch["pixel_values"])
        return {"class_scores": class_scores}

    def state_dict(self):
        return self.backbone.state_dict()


class BackboneFinetuneLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, model_outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]):
        ce_loss = self.loss(model_outputs["class_scores"], batch["class_ids"])
        return {"l_total": ce_loss}


@torch.no_grad()
def compute_corrects(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]):
    class_scores, class_ids = outputs["class_scores"], batch["class_ids"]
    return torch.sum(torch.argmax(class_scores.data, dim=-1) == class_ids.data).item()


def load_backbone_for_ft(
    name: str, num_classes: int, lr: float, step_size: int, gamma: float
) -> tuple[nn.Module, nn.Module, optim.Optimizer, lr_scheduler.LRScheduler]:
    net = BackBone(name=name, num_classes=num_classes)
    loss_fn = BackboneFinetuneLoss()

    optimizer = optim.AdamW(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
    return net, loss_fn, optimizer, scheduler
