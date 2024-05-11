import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as t
from torch.utils.data import Dataset
import torchvision.transforms.functional as f
from .constants import (
    SELECTED_CONCEPTS,
    SELECTED_CONCEPTS_V2,
    fine_grained_parts,
    coarse_grained_parts,
    fine2coarse,
)


class CUBDataset(Dataset):
    def __init__(
        self, dataset_dir: str, num_attrs: int, split="train", transforms=None
    ) -> None:
        super().__init__()
        self.split = split
        self.dataset_dir = dataset_dir
        file_path_df = pd.read_csv(
            os.path.join(dataset_dir, "CUB_200_2011", "images.txt"),
            sep=" ",
            header=None,
            names=["image_id", "file_path"],
        )
        img_class_df = pd.read_csv(
            os.path.join(dataset_dir, "CUB_200_2011", "image_class_labels.txt"),
            sep=" ",
            header=None,
            names=["image_id", "class_id"],
        )
        train_test_split_df = pd.read_csv(
            os.path.join(dataset_dir, "CUB_200_2011", "train_test_split.txt"),
            sep=" ",
            header=None,
            names=["image_id", "is_train"],
        )
        class_names_df = pd.read_csv(
            os.path.join(dataset_dir, "CUB_200_2011", "classes.txt"),
            sep=" ",
            header=None,
            names=["class_id", "class_name"],
        )

        class_attr_labels = np.loadtxt(
            os.path.join(
                dataset_dir,
                "CUB_200_2011",
                "attributes",
                "class_attribute_labels_continuous.txt",
            )
        )
        attribute_df = pd.read_csv(
            "datasets/CUB/attributes.txt",
            sep=" ",
            header=None,
            names=["attribute_id", "attribute_name"],
        ).drop(columns=["attribute_id"])
        attribute_df = attribute_df.iloc[SELECTED_CONCEPTS_V2]
        part_pattern = "|".join(fine_grained_parts)
        attribute_df["part_name"] = attribute_df["attribute_name"].str.extract(
            f"({part_pattern})"
        )
        attribute_df["part_name"] = attribute_df["part_name"].replace(fine2coarse)
        attribute_df["part_id"] = attribute_df["part_name"].map(
            coarse_grained_parts.index
        )
        self.attribute_df = attribute_df.reset_index(drop=True)

        assert num_attrs in [107, 112, 312]
        if num_attrs == 107:
            self.class_attr_labels = class_attr_labels[:, SELECTED_CONCEPTS_V2] / 100
        elif num_attrs == 112:
            self.class_attr_labels = class_attr_labels[:, SELECTED_CONCEPTS] / 100
        else:
            self.class_attr_labels = class_attr_labels

        main_df = file_path_df.merge(img_class_df, on="image_id").merge(
            train_test_split_df, on="image_id"
        )
        main_df["image_id"] -= 1
        main_df["class_id"] -= 1

        train_mask = main_df["is_train"] == 1
        val_mask = ~train_mask
        train_img_ids = main_df.loc[train_mask, "image_id"].unique()
        val_img_ids = main_df.loc[val_mask, "image_id"].unique()

        self.main_df = main_df.set_index("image_id")
        self.img_ids = {"train": train_img_ids, "val": val_img_ids}

        self.class_names = (
            class_names_df["class_name"]
            .str.split(".")
            .str[-1]
            .replace("_", " ", regex=True)
            .to_list()
        )
        self.transforms = transforms

    @property
    def class_attr_embs(self):
        return torch.tensor(self.class_attr_labels, dtype=torch.float32)

    @property
    def attr_groups(self):
        return torch.tensor(self.attribute_df["part_id"].to_numpy())

    def __len__(self):
        return len(self.img_ids[self.split])

    def __getitem__(self, idx):
        img_id = self.img_ids[self.split][idx]

        file_path, class_id, _ = self.main_df.iloc[img_id]

        image = Image.open(
            os.path.join(self.dataset_dir, "CUB_200_2011", "images", file_path)
        ).convert("RGB")
        class_attrs = self.class_attr_labels[class_id]
        # attr_part_groups = self.attribute_df['part_id'].to_numpy()

        if self.transforms:
            pixel_values = self.transforms(image)
        else:
            pixel_values = f.pil_to_tensor(image)

        return {
            "image_ids": img_id,
            "pixel_values": pixel_values,
            "class_ids": torch.tensor(class_id, dtype=torch.long),
            "attr_scores": torch.tensor(class_attrs, dtype=torch.float32),
        }


def get_cub_transforms(resolution: int = 448):
    train_transforms = t.Compose(
        [
            t.Resize(size=resolution, antialias=True),
            t.RandomHorizontalFlip(),
            t.ColorJitter(0.1),
            t.RandomAffine(degrees=90, translate=(0.2, 0.2), scale=(0.8, 1.2)),
            t.RandomCrop(resolution),
            t.ToTensor(),
        ]
    )
    test_transforms = t.Compose(
        [
            t.Resize(size=resolution, antialias=True),
            t.CenterCrop(size=resolution),
            t.ToTensor(),
        ]
    )

    return train_transforms, test_transforms


# p = Augmentor.Pipeline()
# p.rotate(probability=0.4, max_left_rotation=10, max_right_rotation=10)
# p.shear(probability=0.4, max_shear_left=10, max_shear_right=10)
# p.random_distortion(probability=0.4, grid_height=16, grid_width=16, magnitude=8)
# p.skew(probability=0.4)
