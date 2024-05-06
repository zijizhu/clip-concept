import os
import sys
import clip
import json
import torch
import logging
import argparse
import Augmentor
import numpy as np
from torch import nn
from tqdm import tqdm
from pathlib import Path
from clip.model import CLIP
from torchinfo import summary
from datetime import datetime
import torch.nn.functional as F
import torchvision.transforms as T
from lightning import seed_everything
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from data.cub import CUBDatasetSimple
from apn import APN, decorrelation_loss, compactness_loss


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                writer: SummaryWriter, dataset_size: int, epoch: int,
                device: torch.device, logger: logging.Logger):
    running_ad_loss = 0
    running_cpt_loss = 0
    running_local_loss = 0
    running_global_loss = 0
    running_total_loss = 0
    running_corrects = 0

    for batch in tqdm(dataloader):
        images, class_tgts, attr_tgts = batch
        images, class_tgts, attr_tgts = images.to(device), class_tgts.to(device), attr_tgts.to(device)
        global_logits, local_logits, attn_maps, prototypes, max_logit_coords = model(images)

        l_ad = decorrelation_loss(prototypes, model.attr_groups)
        l_cpt = compactness_loss(attn_maps, max_logit_coords)
        l_local = F.mse_loss(local_logits, attr_tgts)
        l_global = F.cross_entropy(global_logits, class_tgts)

        total_loss = l_global + 0.1 * l_local + 0.01 * l_cpt + 0.2 * l_ad

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_ad_loss += running_ad_loss * len(images)
        running_cpt_loss += running_cpt_loss * len(images)
        running_local_loss += running_local_loss * len(images)
        running_global_loss += running_global_loss * len(images)
        running_total_loss += running_total_loss * len(images)
        running_corrects += torch.sum(torch.argmax(global_logits.data, dim=-1) == class_tgts.data).item()

    # Log running losses
    ad_loss_avg = running_ad_loss / dataset_size
    cpt_loss_avg = running_cpt_loss / dataset_size
    local_loss_avg = running_local_loss / dataset_size
    global_loss_avg = running_global_loss / dataset_size
    total_loss_avg = running_total_loss / dataset_size
    epoch_acc = running_corrects / dataset_size

    writer.add_scalar(f'Loss/train/ad', ad_loss_avg, epoch)
    writer.add_scalar(f'Loss/train/cpt', cpt_loss_avg, epoch)
    writer.add_scalar(f'Loss/train/local',local_loss_avg, epoch)
    writer.add_scalar(f'Loss/train/global', global_loss_avg, epoch)
    writer.add_scalar(f'Acc/train/total', total_loss_avg, epoch)
    logger.info(f'EPOCH {epoch} Train ad Loss: {ad_loss_avg:.4f}')
    logger.info(f'EPOCH {epoch} Train cpt Loss: {cpt_loss_avg:.4f}')
    logger.info(f'EPOCH {epoch} Train local Loss: {local_loss_avg:.4f}')
    logger.info(f'EPOCH {epoch} Train global Loss: {global_loss_avg:.4f}')
    logger.info(f'EPOCH {epoch} Train total Loss: {total_loss_avg:.4f}')
    logger.info(f'EPOCH {epoch} Train Acc: {epoch_acc:.4f}')


@torch.no_grad() 
def val_epoch(model, dataloader: DataLoader, writer: SummaryWriter,
              dataset_size: int, epoch: int, device: torch.device, logger: logging.Logger):
    running_corrects = 0

    for batch in tqdm(dataloader):
        images, class_tgts, attr_tgts = batch
        images, class_tgts, attr_tgts = images.to(device), class_tgts.to(device), attr_tgts.to(device)
        global_logits, local_logits, attn_maps, prototypes, max_logit_coords = model(images)

        running_corrects += torch.sum(torch.argmax(global_logits.data, dim=-1) == class_tgts.data).item()

    # Log running losses
    epoch_acc = running_corrects / dataset_size

    writer.add_scalar(f'Acc/val', epoch_acc, epoch)
    logger.info(f'EPOCH {epoch} Val Acc: {epoch_acc:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLIPTuning')
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, choices=['CUB', 'CARS'], required=True)

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', default=1e-6, type=float)
    parser.add_argument('--batch_size', default=16, type=int)

    args = parser.parse_args()
    print(args)

    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log_dir = os.path.join(f'{args.dataset}_runs', datetime.now().strftime('%Y-%m-%d_%H-%M'))
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(log_dir, 'hparams.json'), 'w+') as fp:
        json.dump(vars(args), fp=fp, indent=4)

    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text('Dataset', args.dataset)
    writer.add_text('Device', str(device))
    writer.add_text('Batch size', str(args.batch_size))
    writer.add_text('Epochs', str(args.epochs))
    writer.add_text('Seed', str(args.seed))

    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[
                            logging.FileHandler(os.path.join(log_dir, 'train.log')),
                            logging.StreamHandler(sys.stdout)
                        ])
    logger = logging.getLogger(__name__)
    
    # clip_model, clip_preprocess = clip.load('ViT-B/16', device=torch.device('cpu'))

    if args.dataset == 'CUB':
        # p = Augmentor.Pipeline()
        # p.rotate(probability=0.4, max_left_rotation=10, max_right_rotation=10)
        # p.shear(probability=0.4, max_shear_left=10, max_shear_right=10)
        # p.random_distortion(probability=0.4, grid_height=16, grid_width=16, magnitude=8)
        # p.skew(probability=0.4)
        # train_transforms = T.Compose([T.Resize(448), T.CenterCrop(448), p.torch_transform(), T.RandomHorizontalFlip(p=0.4), T.PILToTensor()])
        # test_transforms = T.Compose([T.Resize(448), T.CenterCrop(448), T.PILToTensor()])
        train_transforms = T.Compose([
            T.Resize(size=448, antialias=True),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.1),
            T.RandomAffine(degrees=90, translate=(0.2, 0.2), scale=(0.8, 1.2)),
            T.RandomCrop(448),
            T.ToTensor()
        ])
        test_transforms = T.Compose([
            T.Resize(size=448, antialias=True),
            T.CenterCrop(size=448),
            T.ToTensor()
        ])

        dataset_train = CUBDatasetSimple(os.path.join(args.dataset_dir, 'CUB'), split='train', transforms=train_transforms)
        dataset_val = CUBDatasetSimple(os.path.join(args.dataset_dir, 'CUB'), split='val', transforms=test_transforms)
        dataloader_train = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=8)
        dataloader_val = DataLoader(dataset=dataset_val, batch_size=args.batch_size, shuffle=True, num_workers=8)

        k, attr_class_map, attr_groups = 107, dataset_train.attr_class_map, dataset_train.attr_groups
    elif args.dataset == 'CARS':
        raise NotImplementedError
        # def collate_fn(batch):
        #     image_list, label_list = list(zip(*batch))
        #     return image_list, torch.tensor(label_list)

        # dataset_train = StanfordCars(root=os.path.join(args.dataset_dir, 'CARS'), split='train', download=True)
        # dataset_val = StanfordCars(root=os.path.join(args.dataset_dir, 'CARS'), split='test', download=True)
        # dataloader_train = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        # dataloader_val = DataLoader(dataset=dataset_val, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    else:
        raise NotImplementedError

    model = APN(attr_class_map, k=k, attr_groups=attr_groups)
    print(summary(model))
    model.to(device)

    # Classification using prototypes
    logger.info('Start training...')
    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.5, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.9)

    model.train()
    for epoch in range(args.epochs):
        train_epoch(model=model, dataloader=dataloader_train, optimizer=optimizer,
                    writer=writer, dataset_size=len(dataset_train),
                    device=device, epoch=epoch, logger=logger)
        val_epoch(model=model, dataloader=dataloader_val, writer=writer, dataset_size=len(dataset_val),
                  device=device, epoch=epoch, logger=logger)
        torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                   os.path.join(log_dir, 'checkpoint.pt'))
        scheduler.step()
