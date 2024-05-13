import os
import clip
import json
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as f
from torch.utils.data import DataLoader

from data.cub.cub_dataset import CUBDataset


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_model, clip_preprocess = clip.load('ViT-B/16', device=device)

    dataset_train = CUBDataset(os.path.join('datasets', 'CUB'), num_attrs=312,
                               split='train', transforms=clip_preprocess)
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=32,
                                  shuffle=False, num_workers=8)

    with open('concepts.txt', 'r') as fp:
        concepts = fp.read().splitlines()
    concepts_tokenized = clip.tokenize(concepts)

    class_ids = []
    all_similarities = []
    all_image_features = []

    with torch.no_grad():
        text_features = clip_model.encode_text(concepts_tokenized.to(device))
        text_features_norm = f.normalize(text_features, dim=-1)

        for i, batch in enumerate(dataloader_train):
            image_features = clip_model.encode_image(batch['pixel_values'].to(device))
            image_features_norm = f.normalize(image_features, dim=-1)
            similarities = image_features_norm @ text_features_norm.T

            all_image_features.append(image_features_norm)
            all_similarities.append(similarities)
            class_ids.append(batch['class_ids'])

    class_ids = torch.cat(class_ids)
    similarities = torch.cat(all_similarities)
    image_features_norm = torch.cat(all_image_features)
    np.savez('checkpoints/clip_inference',
             class_ids=class_ids.cpu().numpy(),
             similarities=similarities.cpu().numpy(),
             image_features_norm=image_features_norm.cpu().numpy(),
             text_features_norm=text_features_norm.cpu().numpy())
