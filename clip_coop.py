
import os
import sys
import clip
import json
import torch
import logging
import argparse
import numpy as np
from torch import nn
from tqdm import tqdm
from pathlib import Path
from clip.model import CLIP
from torchinfo import summary
from datetime import datetime
import torch.nn.functional as F
from lightning import seed_everything
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from data.cub import CUBDatasetSimple

class TextEncoder(nn.Module):
    def __init__(self, clip_model: CLIP):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, soft_prompt_emb, prompt_token_ids):
        x = soft_prompt_emb + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # shape: [num_classes, num_tokens, dim] -> [num_tokens, num_classes, dim]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # shape: [num_tokens, num_classes, dim] -> [num_classes, num_tokens, dim]
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), prompt_token_ids.argmax(dim=-1)] @ self.text_projection

        return x


class SoftPrompt(nn.Module):
    def __init__(self, class_names, clip_model: CLIP, tokenizer_fn, nctx=16) -> None:
        super().__init__()

        self.nctx = nctx
        dtype = clip_model.dtype
        dim, = clip_model.ln_final.weight.shape
        self.ctx = nn.Parameter(torch.normal(mean=0, std=0.02, size=(nctx, dim), dtype=dtype))

        self.num_classes = len(class_names)
        dummy_texts = " ".join(['X'] * nctx)

        class_names = [name.replace("_", " ") for name in class_names]
        prompt_texts = [f'{dummy_texts} {name}.' for name in class_names]

        self.prompt_token_ids = torch.cat([tokenizer_fn(p) for p in prompt_texts])
        with torch.no_grad():
            embeddings = clip_model.token_embedding(self.prompt_token_ids).to(dtype)

        self.register_buffer("prefix_emb", embeddings[:, :1, :])  # SOT, shape: [num_classes, 1, dim]
        self.register_buffer("suffix_emb", embeddings[:, 1+nctx:, :])  # Class tokens and EOT, shape: [num_classes, *, dim]
    
    def forward(self):
        ctx = self.ctx
        ctx = ctx.unsqueeze(0).expand(self.num_classes, -1, -1)  # shape: [num_classes, nctx, dim]
        soft_prompts = torch.cat([self.prefix_emb, ctx, self.suffix_emb], dim=1)
        return soft_prompts, self.prompt_token_ids


class CLIPWithSoftPrompt(nn.Module):
    def __init__(self, class_names, clip_model: CLIP, clip_tokenize_fn):
        super().__init__()
        self.create_soft_prompt = SoftPrompt(class_names, clip_model, clip_tokenize_fn)
        self.visual_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        for param in self.visual_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def forward(self, image):
        image_features = self.visual_encoder(image.to(self.dtype))

        soft_prompt_emb, prompt_token_ids = self.create_soft_prompt()
        text_features = self.text_encoder(soft_prompt_emb, prompt_token_ids)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                writer: SummaryWriter, dataset_size: int, epoch: int,
                device: torch.device, logger: logging.Logger):
    running_loss = 0
    running_corrects = 0

    for batch in tqdm(dataloader):
        images, targets = batch
        targets = targets.to(device)
        logits = model(images.to(device))

        loss = F.cross_entropy(logits, targets)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss * len(images)
        running_corrects += torch.sum(torch.argmax(logits.data, dim=-1) == targets.data).item()

    # Log running losses
    loss_avg = running_loss / dataset_size
    epoch_acc = running_corrects / dataset_size

    writer.add_scalar(f'Loss/train', loss_avg, epoch)
    writer.add_scalar(f'Acc/train', epoch_acc, epoch)
    logger.info(f'EPOCH {epoch} Train Loss: {loss_avg:.4f}')
    logger.info(f'EPOCH {epoch} Train Aac: {epoch_acc:.4f}')


@torch.no_grad() 
def val_epoch(model, dataloader: DataLoader, writer: SummaryWriter,
              dataset_size: int, epoch: int, device: torch.device, logger: logging.Logger):
    running_corrects = 0

    for batch in tqdm(dataloader):
        images, targets = batch
        targets = targets.to(device)
        logits = model(images.to(device))

        running_corrects += torch.sum(torch.argmax(logits.data, dim=-1) == targets.data).item()

    # Log running losses
    epoch_acc = running_corrects / dataset_size

    writer.add_scalar(f'Acc/val', epoch_acc, epoch)
    logger.info(f'EPOCH {epoch} Val Aac: {epoch_acc:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLIPTuning')
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, choices=['CUB', 'CARS'], required=True)

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', default=2e-3, type=float)
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
    writer.add_text('Learning rate', str(args.lr))
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
    
    clip_model, clip_preprocess = clip.load('ViT-B/16', device=device)

    if args.dataset == 'CUB':
        # def collate_fn(batch):
        #     image_list, label_list = list(zip(*batch))
        #     return image_list, torch.stack(label_list)

        dataset_train = CUBDatasetSimple(os.path.join(args.dataset_dir, 'CUB'), split='train', transforms=clip_preprocess)
        dataset_val = CUBDatasetSimple(os.path.join(args.dataset_dir, 'CUB'), split='val', transforms=clip_preprocess)
        dataloader_train = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True)
        dataloader_val = DataLoader(dataset=dataset_val, batch_size=args.batch_size, shuffle=True)

        class_names = dataset_train.class_names
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

    model = CLIPWithSoftPrompt(class_names, clip_model, clip.tokenize)
    print(summary(model))

    # Classification using prototypes
    logger.info('Start training...')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.5)

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
