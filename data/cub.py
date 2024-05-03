import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

class CUBDatasetSimple(Dataset):
    def __init__(self, dataset_dir: str, split='train', transforms=None) -> None:
        super().__init__()
        self.split = split
        self.dataset_dir = dataset_dir
        file_path_df = pd.read_csv(os.path.join(dataset_dir, 'CUB_200_2011', 'images.txt'),
                                    sep=' ', header=None, names=['image_id', 'file_path'])
        img_class_df = pd.read_csv(os.path.join(dataset_dir, 'CUB_200_2011', 'image_class_labels.txt'),
                                    sep=' ', header=None, names=['image_id', 'class_id'])
        train_test_split_df = pd.read_csv(os.path.join(dataset_dir, 'CUB_200_2011', 'train_test_split.txt'),
                                            sep=' ', header=None, names=['image_id', 'is_train'])
        class_names_df = pd.read_csv(os.path.join(dataset_dir, 'CUB_200_2011', 'classes.txt'),
                                     sep=' ', header=None, names=['class_id', 'class_name'])

        main_df = (file_path_df
                   .merge(img_class_df, on='image_id')
                   .merge(train_test_split_df, on='image_id'))
        main_df['image_id'] -= 1
        main_df['class_id'] -= 1
        
        train_mask = main_df['is_train'] == 1
        val_mask = ~train_mask
        train_img_ids= main_df.loc[train_mask, 'image_id'].unique()
        val_img_ids = main_df.loc[val_mask, 'image_id'].unique()

        self.main_df = main_df.set_index('image_id')
        self.img_ids = {
            'train': train_img_ids,
            'val': val_img_ids
        }

        self.class_names = class_names_df['class_name'].str.split('.').str[-1].replace('_', ' ', regex=True).to_list()
        self.transforms = transforms

    def __len__(self):
        return len(self.img_ids[self.split])

    def __getitem__(self, idx):
        img_id = self.img_ids[self.split][idx]

        file_path, class_id, _ = self.main_df.iloc[img_id]

        image = Image.open(os.path.join(self.dataset_dir, 'CUB_200_2011', 'images', file_path)).convert('RGB')

        if self.transforms:
            image_tensor = self.transforms(image)
        else:
            image_tensor = F.pil_to_tensor(image)

        return image_tensor, torch.tensor(class_id, dtype=torch.long)
