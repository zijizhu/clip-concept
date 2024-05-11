import os
import sys
import torch
import logging
import argparse
from torch import nn
from tqdm import tqdm
from pathlib import Path
from typing import Callable
from datetime import datetime
from omegaconf import OmegaConf
from lightning import seed_everything
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from data.cub.cub_dataset import CUBDataset, get_cub_transforms
from apn import load_backbone_for_ft, load_apn, compute_corrects


def train_epoch(model: nn.Module, loss_fn: nn.Module, loss_keys: list[str], acc_fn: nn.Module | Callable,
                dataloader: DataLoader, optimizer: torch.optim.Optimizer, writer: SummaryWriter,
                dataset_size: int, epoch: int, batch_size: int, device: torch.device, logger: logging.Logger):

    running_losses = {k: 0 for k in loss_keys}
    running_corrects = 0

    for batch_inputs in tqdm(dataloader):
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        outputs = model(batch_inputs)
        loss_dict = loss_fn(outputs, batch_inputs)  # type: dict[str, torch.Tensor]

        total_loss = loss_dict['l_total']

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        for loss_name, loss in loss_dict.items():
            running_losses[loss_name] += loss.item() * batch_size

        running_corrects += acc_fn(outputs, batch_inputs)

    # Log metrics
    for loss_name, loss in running_losses.items():
        loss_avg = loss / dataset_size
        writer.add_scalar(f'Loss/train/{loss_name}', loss_avg, epoch)
        logger.info(f'EPOCH {epoch} Train {loss_name}: {loss_avg:.4f}')

    epoch_acc = running_corrects / dataset_size
    writer.add_scalar('Acc/train', epoch_acc, epoch)
    logger.info(f'EPOCH {epoch} Train Acc: {epoch_acc:.4f}')


@torch.no_grad()
def val_epoch(model: nn.Module, acc_fn: nn.Module | Callable, dataloader: DataLoader, writer: SummaryWriter,
              dataset_size: int, epoch: int, batch_size: int, device: torch.device, logger: logging.Logger):

    running_corrects = 0

    for batch_inputs in tqdm(dataloader):
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        outputs = model(batch_inputs)

        running_corrects += acc_fn(outputs, batch_inputs)

    epoch_acc = running_corrects / dataset_size
    writer.add_scalar('Acc/val', epoch_acc, epoch)
    logger.info(f'EPOCH {epoch} Val Acc: {epoch_acc:.4f}')


def main():
    parser = argparse.ArgumentParser(description='Training Script')
    parser.add_argument('-c', '--config_path', type=str, required=True)
    parser.add_argument('-o', '--options', type=str, nargs='+')

    args = parser.parse_args()
    config_path = Path(args.config_path)
    base_cfg = OmegaConf.load(config_path)
    if args.options:
        cli_cfg = OmegaConf.from_dotlist(args.options)
        cfg = OmegaConf.merge(base_cfg, cli_cfg)
    else:
        cfg = base_cfg

    seed_everything(cfg.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    experiment_name = config_path.stem
    print('Experiment name:', experiment_name)
    print('Hyperparameters:')
    print(OmegaConf.to_yaml(cfg))
    print('Device:', device)

    #################
    # Setup logging #
    #################

    log_dir = os.path.join('logs', f'{datetime.now().strftime("%Y-%m-%d_%H-%M")}_{experiment_name}')
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(log_dir, 'hparams.yaml'), 'w+') as fp:
        OmegaConf.save(config=OmegaConf.merge(OmegaConf.create({'NAME': experiment_name}), cfg), f=fp.name)

    summary_writer = SummaryWriter(log_dir=log_dir)
    summary_writer.add_text('Model', cfg.MODEL.NAME)
    summary_writer.add_text('Dataset', cfg.DATASET.NAME)
    summary_writer.add_text('Batch size', str(cfg.OPTIM.BATCH_SIZE))
    summary_writer.add_text('Epochs', str(cfg.OPTIM.EPOCHS))
    summary_writer.add_text('Seed', str(cfg.SEED))
    summary_writer.add_text('Device', str(device))

    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[
                            logging.FileHandler(os.path.join(log_dir, 'train.log')),
                            logging.StreamHandler(sys.stdout)
                        ])
    logger = logging.getLogger(__name__)

    ###########################
    # Setup dataset and model #
    ###########################

    if cfg.DATASET.NAME == 'CUB':
        train_transforms, test_transforms = get_cub_transforms(resolution=cfg.MODEL.IMAGE_SIZE)
        dataset_train = CUBDataset(os.path.join(cfg.DATASET.ROOT_DIR, 'CUB'), num_attrs=cfg.DATASET.NUM_ATTRS,
                                   split='train', transforms=train_transforms)
        dataset_val = CUBDataset(os.path.join(cfg.DATASET.ROOT_DIR, 'CUB'), num_attrs=cfg.DATASET.NUM_ATTRS,
                                 split='val', transforms=test_transforms)
        dataloader_train = DataLoader(dataset=dataset_train, batch_size=cfg.OPTIM.BATCH_SIZE,
                                      shuffle=True, num_workers=8)
        dataloader_val = DataLoader(dataset=dataset_val, batch_size=cfg.OPTIM.BATCH_SIZE,
                                    shuffle=True, num_workers=8)
    elif cfg.DATASET.NAME == 'CARS':
        raise NotImplementedError
    else:
        raise NotImplementedError

    if 'ft' in experiment_name:
        net, loss_fn, optimizer, scheduler = load_backbone_for_ft(name=cfg.MODEL.NAME,
                                                                  num_classes=cfg.DATASET.NUM_CLASSES,
                                                                  lr=cfg.OPTIM.LR,
                                                                  step_size=cfg.OPTIM.STEP_SIZE,
                                                                  gamma=cfg.OPTIM.GAMMA)
        losses = ['l_total']  # Only need cross entropy for fine-tuning backbone
    else:
        class_attr_embs = dataset_train.class_attr_embs
        net, loss_fn, optimizer, scheduler = load_apn(num_classes=cfg.DATASET.NUM_CLASSES,
                                                      num_attrs=cfg.DATASET.NUM_ATTRS,
                                                      dist=cfg.MODEL.DIST,
                                                      class_attr_embs=class_attr_embs,
                                                      backbone_name=cfg.MODEL.BACKBONE.NAME,
                                                      backbone_weight_path=cfg.MODEL.BACKBONE.CKPT_PATH,
                                                      loss_coef_dict=dict(cfg.MODEL.LOSSES),
                                                      lr=cfg.OPTIM.LR,
                                                      step_size=cfg.OPTIM.STEP_SIZE,
                                                      gamma=cfg.OPTIM.GAMMA)
        losses = list(name.lower() for name in cfg.MODEL.LOSSES.keys()) + ['l_total']

    #################
    # Training loop #
    #################

    logger.info('Start training...')
    net.to(device)
    net.train()
    for epoch in range(cfg.OPTIM.EPOCHS):
        train_epoch(model=net, loss_fn=loss_fn, loss_keys=losses, acc_fn=compute_corrects,
                    dataloader=dataloader_train, optimizer=optimizer, writer=summary_writer,
                    batch_size=cfg.OPTIM.BATCH_SIZE, dataset_size=len(dataset_train),
                    device=device, epoch=epoch, logger=logger)

        val_epoch(model=net, acc_fn=compute_corrects, dataloader=dataloader_val, writer=summary_writer,
                  dataset_size=len(dataset_val), batch_size=cfg.OPTIM.BATCH_SIZE,
                  device=device, epoch=epoch, logger=logger)

        torch.save({k: v.cpu() for k, v in net.state_dict().items()},
                   os.path.join(log_dir, f'{experiment_name}.pt'))

        if scheduler:
            scheduler.step()
    logger.info('DONE!')


if __name__ == '__main__':
    main()
