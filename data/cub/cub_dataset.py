import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as t
import torchvision.transforms.functional as f
from PIL import Image
from torch.utils.data import Dataset

from .constants import PART_GROUPS, SELECTED_CONCEPTS


class CUBDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        num_attrs: int = 312,
        split: str = "train",
        groups: str | None = None,
        transforms: t.Compose | None = None,
    ) -> None:
        super().__init__()
        self.split = split
        self.dataset_dir = dataset_dir

        #####################################################
        # Load dataframes that store information of samples #
        #####################################################

        file_paths_df = pd.read_csv(
            os.path.join(dataset_dir, "CUB_200_2011", "images.txt"),
            sep=" ",
            header=None,
            names=["image_id", "filename"],
        )
        image_labels_df = pd.read_csv(
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

        main_df = file_paths_df.merge(image_labels_df, on="image_id").merge(
            train_test_split_df, on="image_id"
        )

        main_df["image_id"] -= 1
        main_df["class_id"] -= 1

        train_mask = main_df["is_train"] == 1
        val_mask = ~train_mask
        train_image_ids = main_df.loc[train_mask, "image_id"].unique()
        val_image_ids = main_df.loc[val_mask, "image_id"].unique()

        self.main_df = main_df.set_index("image_id")
        self.image_ids = {"train": train_image_ids, "val": val_image_ids}

        ###############################
        # Load and process attributes #
        ###############################

        # Load attribute names
        attr_df = pd.read_csv(
            os.path.join(dataset_dir, "attributes.txt"),
            sep=' ',
            usecols=[1],
            header=None,
            names=['attribute']
        )
        attr_df['attribute'] = attr_df['attribute'].str.replace('_', ' ', regex=True)
        attr_df[['attribute_group', 'value']] = attr_df['attribute'].str.split('::', n=1, expand=True)

        if num_attrs == 112:
            attr_df = attr_df.iloc[SELECTED_CONCEPTS]

        self.attributes_df = attr_df.reset_index(drop=True)

        # Load attribute vectors
        assert num_attrs in [112, 312]
        attribute_vectors = np.loadtxt(
            os.path.join(
                dataset_dir,
                "CUB_200_2011",
                "attributes",
                "class_attribute_labels_continuous.txt",
            )
        )
        
        if num_attrs == 112:
            attribute_vectors = attribute_vectors[:, SELECTED_CONCEPTS]

        # Normalize attribute vectors
        attribute_vectors /= np.linalg.norm(attribute_vectors, ord=2, axis=-1, keepdims=True)
        self.attribute_vectors = attribute_vectors  # type: np.ndarray

        #####################################################
        # Load and process class names in a readable format #
        #####################################################

        class_names_df = pd.read_csv(
            os.path.join(dataset_dir, "CUB_200_2011", "classes.txt"),
            sep=" ",
            header=None,
            names=["class_id", "class_name"],
        )

        self.class_names = (
            class_names_df["class_name"]
            .str
            .split(".")
            .str[-1]
            .replace("_", " ", regex=True)
            .to_list()
        )

        #################################
        # Load part or attribute groups #
        #################################
        assert groups in ["parts", "attributes"]
        self.groups = groups
        if groups == 'parts':
            assert num_attrs == 312
            self.group_names = sorted(PART_GROUPS.keys())
            group_indices = np.zeros(312, dtype=int)
            for i, name in enumerate(self.group_names):
                attr_indices = PART_GROUPS[name]
                group_indices[attr_indices] = i
            self.attribute_group_indices = group_indices
        elif groups == 'attributes':
            self.group_names = sorted(self.attributes_df['attribute_group'].unique())
            self.attribute_group_indices = (self.attributes_df['attribute_group']
                                            .map(self.group_names.index)
                                            .to_numpy())
        else:
            raise NotImplementedError


        self.transforms = transforms

    @property
    def attribute_group_indices_pt(self):
        return torch.tensor(self.attribute_group_indices, dtype=torch.long)

    @property
    def attribute_vectors_pt(self):
        return torch.tensor(self.attribute_vectors, dtype=torch.float32)

    def get_topk_attributes(
        self, class_id: torch.Tensor | int, k: int = 5, pred_scores: torch.Tensor | None = None
    ) -> pd.DataFrame:
        """Given a class id, return k attributes with highest (ground truth) scores of this class"""
        if isinstance(class_id, torch.Tensor):
            class_id = class_id.item()
        values, indices = torch.topk(self.attribute_vectors_pt[class_id], k=k)
        top_attributes_df = self.attributes_df[indices.numpy()]
        top_attributes_df["ground_truth_scores"] = values.numpy()
        if pred_scores:
            top_attributes_df["predicted_scores"] = pred_scores.numpy()
        return top_attributes_df

    def __len__(self):
        return len(self.image_ids[self.split])

    def __getitem__(self, idx):
        image_id = self.image_ids[self.split][idx]

        filename, class_id, _ = self.main_df.iloc[image_id]

        path_to_image = os.path.join(self.dataset_dir, "CUB_200_2011", "images", filename)
        image = Image.open(path_to_image).convert("RGB")

        attr_scores = self.attribute_vectors_pt[class_id]

        pixel_values = self.transforms(image) if self.transforms else f.pil_to_tensor(image)

        return {
            "image_ids": image_id,
            "pixel_values": pixel_values,
            "class_ids": torch.tensor(class_id, dtype=torch.long),
            "attr_scores": attr_scores.clone(),
        }


def get_transforms_part_discovery(resolution: int = 448):
    """A set of transforms used in https://github.com/robertdvdk/part_detection"""
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


def get_transforms_resnet101():
    """A set of transforms for standard resnet101 training from torchvision, reference:
    https://pytorch.org/vision/main/models/generated/torchvision.models.resnet101.html"""
    train_transforms = t.Compose(
        [
            t.Resize(size=232, interpolation=t.InterpolationMode.BILINEAR),
            t.CenterCrop(size=224),
            t.ToTensor(),
            t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_transforms = t.Compose(
        [
            t.Resize(size=232, interpolation=t.InterpolationMode.BILINEAR),
            t.CenterCrop(size=224),
            t.ToTensor(),
            t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transforms, test_transforms
