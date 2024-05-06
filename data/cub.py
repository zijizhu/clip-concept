import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


SELECTED_CONCEPTS = [1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45,
                     50, 51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91, 93,
                     99, 101, 106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149,
                     151, 152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 181, 183, 187, 188,
                     193, 194, 196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225,
                     235, 236, 238, 239, 240, 242, 243, 244, 249, 253, 254, 259, 260, 262, 268,
                     274, 277, 283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311]


SELECTED_CONCEPTS_V2 = [1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45,
                        50, 51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91, 93,
                        99, 101, 106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149,
                        151, 152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 181, 183, 187, 188,
                        193, 194, 196, 198, 202, 203, 208, 209, 211, 212, 213, 236, 238, 239, 240,
                        242, 243, 244, 249, 253, 254, 259, 260, 262, 268, 274, 277, 283, 289, 292,
                        293, 294, 298, 299, 304, 305, 308, 309, 310, 311]


fine_grained_parts = ['bill', 'upperpart', 'underpart', 'head', 'back', 'beak', 'belly', 'breast',
                      'crown', 'eye', 'forehead', 'leg', 'nape', 'tail', 'throat', 'wing', 'primary']
coarse_grained_parts = ['head', 'back', 'belly', 'breast', 'leg', 'tail', 'wing']
fine2coarse =  {
    'bill': 'head',
    'upperpart': 'breast',
    'underpart': 'belly',
    'head': 'head',
    'back': 'back',
    'beak': 'head',
    'belly': 'belly',
    'breast': 'breast',
    'crown': 'head',
    'eye': 'head',
    'forehead': 'head',
    'leg': 'leg',
    'nape': 'back',
    'tail': 'tail',
    'throat': 'breast',
    'wing': 'wing',
    'primary': 'breast'
}

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

        class_attr_labels = np.loadtxt(os.path.join(dataset_dir, 'CUB_200_2011', 'attributes',
                                                    'class_attribute_labels_continuous.txt'))
        attribute_df = pd.read_csv('datasets/CUB/attributes.txt',
                                   sep=' ',
                                   header=None,
                                   names=['attribute_id', 'attribute_name']).drop(columns=['attribute_id'])
        attribute_df = attribute_df.iloc[SELECTED_CONCEPTS_V2]
        part_pattern = '|'.join(fine_grained_parts)
        attribute_df['part_name'] = attribute_df['attribute_name'].str.extract(f'({part_pattern})')
        attribute_df['part_name'] = attribute_df['part_name'].replace(fine2coarse)
        attribute_df['part_id'] = attribute_df['part_name'].map(coarse_grained_parts.index)
        self.attribute_df = attribute_df

        self.class_attr_labels = class_attr_labels[:, SELECTED_CONCEPTS_V2] / 100

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
    
    @property
    def attr_class_map(self):
        return torch.tensor(self.class_attr_labels, dtype=torch.float32)
    
    @property
    def attr_groups(self):
        return torch.tensor(self.attribute_df['part_id'].to_numpy())

    def __len__(self):
        return len(self.img_ids[self.split])

    def __getitem__(self, idx):
        img_id = self.img_ids[self.split][idx]

        file_path, class_id, _ = self.main_df.iloc[img_id]

        image = Image.open(os.path.join(self.dataset_dir, 'CUB_200_2011', 'images', file_path)).convert('RGB')
        class_attrs = self.class_attr_labels[class_id]
        attr_part_groups = self.attribute_df['part_id'].to_numpy()

        if self.transforms:
            image_tensor = self.transforms(image)
        else:
            image_tensor = F.pil_to_tensor(image)

        return (image_tensor.to(torch.float32),
                torch.tensor(class_id, dtype=torch.long),
                torch.tensor(class_attrs, dtype=torch.float32))
