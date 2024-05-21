import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
from lightning import seed_everything
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from apn import compute_corrects, load_apn, load_backbone_for_ft
from data.cub.cub_dataset import CUBDataset, get_transforms_part_discovery, get_transforms_resnet101


def train_epoch(
    model: nn.Module,
    loss_fn: nn.Module,
    loss_keys: list[str],
    num_corrects_fn: nn.Module | Callable,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    dataset_size: int,
    epoch: int,
    batch_size: int,
    device: torch.device,
    logger: logging.Logger,
):
    running_losses = {k: 0 for k in loss_keys}
    running_corrects = 0

    for batch_inputs in tqdm(dataloader):
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        outputs = model(batch_inputs)
        loss_dict, total_loss = loss_fn(outputs, batch_inputs)

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        for loss_name, loss in loss_dict.items():
            running_losses[loss_name] += loss * batch_size

        running_corrects += num_corrects_fn(outputs, batch_inputs)

    # Log metrics
    for loss_name, loss in running_losses.items():
        loss_avg = loss / dataset_size
        writer.add_scalar(f"Loss/train/{loss_name}", loss_avg, epoch)
        logger.info(f"EPOCH {epoch} Train {loss_name}: {loss_avg:.4f}")

    epoch_acc = running_corrects / dataset_size
    writer.add_scalar("Acc/train", epoch_acc, epoch)
    logger.info(f"EPOCH {epoch} Train Acc: {epoch_acc:.4f}")


@torch.no_grad()
def val_epoch(
    model: nn.Module,
    num_corrects_fn: nn.Module | Callable,
    dataloader: DataLoader,
    writer: SummaryWriter,
    dataset_size: int,
    epoch: int,
    device: torch.device,
    logger: logging.Logger,
):
    running_corrects = 0

    for batch_inputs in tqdm(dataloader):
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        outputs = model(batch_inputs)

        running_corrects += num_corrects_fn(outputs, batch_inputs)

    epoch_acc = running_corrects / dataset_size
    writer.add_scalar("Acc/val", epoch_acc, epoch)
    logger.info(f"EPOCH {epoch} Val Acc: {epoch_acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("-c", "--config_path", type=str, required=True)
    parser.add_argument("-o", "--options", type=str, nargs="+")

    args = parser.parse_args()
    config_path = Path(args.config_path)
    base_cfg = OmegaConf.load(config_path)
    if args.options:
        cli_cfg = OmegaConf.from_dotlist(args.options)
        cfg = OmegaConf.merge(base_cfg, cli_cfg)
    else:
        cfg = base_cfg

    seed_everything(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment_name = config_path.stem
    print("Experiment name:", experiment_name)
    print("Hyperparameters:")
    print(OmegaConf.to_yaml(cfg))
    print("Device:", device)

    #################
    # Setup logging #
    #################

    log_dir = os.path.join("logs", f'{datetime.now().strftime("%Y-%m-%d_%H-%M")}_{experiment_name}')
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(log_dir, "hparams.yaml"), "w+") as fp:
        OmegaConf.save(OmegaConf.merge(OmegaConf.create({"NAME": experiment_name}), cfg), f=fp.name)

    summary_writer = SummaryWriter(log_dir=log_dir)
    summary_writer.add_text("Model", cfg.MODEL.NAME)
    summary_writer.add_text("Dataset", cfg.DATASET.NAME)
    summary_writer.add_text("Batch size", str(cfg.OPTIM.BATCH_SIZE))
    summary_writer.add_text("Epochs", str(cfg.OPTIM.EPOCHS))
    summary_writer.add_text("Seed", str(cfg.SEED))
    summary_writer.add_text("Device", str(device))

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "train.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(__name__)

    #################################
    # Setup datasets and transforms #
    #################################

    if cfg.DATASET.NAME == "CUB":
        if cfg.DATASET.TRANSFORMS == "resnet101":
            train_transforms, test_transforms = get_transforms_resnet101()
        elif cfg.DATASET.TRANSFORMS == "pasrt_discovery":
            train_transforms, test_transforms = get_transforms_part_discovery()
        else:
            raise NotImplementedError

        num_attrs = cfg.get("DATASET.NUM_ATTRS", 312)
        dataset_train = CUBDataset(
            os.path.join(cfg.DATASET.ROOT_DIR, "CUB"),
            num_attrs=num_attrs,
            split="train",
            transforms=train_transforms,
        )
        dataset_val = CUBDataset(
            os.path.join(cfg.DATASET.ROOT_DIR, "CUB"),
            num_attrs=num_attrs,
            split="val",
            transforms=test_transforms,
        )
        dataloader_train = DataLoader(
            dataset=dataset_train,
            batch_size=cfg.OPTIM.BATCH_SIZE,
            shuffle=True,
            num_workers=8
        )
        dataloader_val = DataLoader(
            dataset=dataset_val,
            batch_size=cfg.OPTIM.BATCH_SIZE,
            shuffle=True,
            num_workers=8
        )
    elif cfg.DATASET.NAME == "CARS":
        raise NotImplementedError
    else:
        raise NotImplementedError

    ##############################
    # Load models and optimizers #
    ##############################

    if "ft" in experiment_name:
        net, loss_fn, optimizer, scheduler = load_backbone_for_ft(
            name=cfg.MODEL.NAME,
            num_classes=cfg.DATASET.NUM_CLASSES,
            lr=cfg.OPTIM.LR,
            step_size=cfg.OPTIM.STEP_SIZE,
            gamma=cfg.OPTIM.GAMMA,
        )
        losses = ["l_total"]
    elif "apn" in experiment_name:
        class_embeddings = dataset_train.attribute_vectors_pt
        net, loss_fn, optimizer, scheduler = load_apn(
            backbone_name=cfg.MODEL.BACKBONE.NAME,
            # backbone_weights_path=cfg.MODEL.BACKBONE.CKPT_PATH,
            class_embeddings=class_embeddings,
            loss_coef_dict=dict(cfg.MODEL.LOSSES),
            dist=cfg.MODEL.DIST,
            lr=cfg.OPTIM.LR,
            betas=(cfg.OPTIM.BETA1, cfg.OPTIM.BETA2),
            step_size=cfg.OPTIM.STEP_SIZE,
            gamma=cfg.OPTIM.GAMMA,
        )
        losses = list(name.lower() for name in cfg.MODEL.LOSSES)
    else:
        raise NotImplementedError

    #################
    # Training loop #
    #################

    logger.info("Start training...")
    net.to(device)
    net.train()
    for epoch in range(cfg.OPTIM.EPOCHS):
        train_epoch(
            model=net,
            loss_fn=loss_fn,
            loss_keys=losses,
            num_corrects_fn=compute_corrects,
            dataloader=dataloader_train,
            optimizer=optimizer,
            writer=summary_writer,
            batch_size=cfg.OPTIM.BATCH_SIZE,
            dataset_size=len(dataset_train),
            device=device,
            epoch=epoch,
            logger=logger,
        )

        val_epoch(
            model=net,
            num_corrects_fn=compute_corrects,
            dataloader=dataloader_val,
            writer=summary_writer,
            dataset_size=len(dataset_val),
            device=device,
            epoch=epoch,
            logger=logger,
        )

        torch.save(
            {k: v.cpu() for k, v in net.state_dict().items()},
            os.path.join(log_dir, f"{experiment_name}.pt"),
        )

        if scheduler:
            scheduler.step()
    logger.info("DONE!")


if __name__ == "__main__":
    main()
