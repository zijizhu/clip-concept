
import os
import random
import Augmentor
from torch.utils.data import DataLoader
import torchvision.transforms as T
from data.cub import CUBDatasetSimple

class CustomTransform:
    def __init__(self, p):
        self.p = p
    
    def __call__(self, image):
        for operation in self.p.operations:
            r = random.uniform(0, 1)
            if r < operation.probability:
                image = operation.perform_operation([image])[0]
        return image

if __name__ == '__main__':
    p = Augmentor.Pipeline()
    p.rotate(probability=0.4, max_left_rotation=10, max_right_rotation=10)
    p.shear(probability=0.4, max_shear_left=10, max_shear_right=10)
    p.random_distortion(probability=0.4, grid_height=16, grid_width=16, magnitude=8)
    p.skew(probability=0.4)
    train_transforms = T.Compose([T.Resize(448), T.CenterCrop(448), CustomTransform(p), T.RandomHorizontalFlip(p=0.4), T.PILToTensor()])
    dataset_train = CUBDatasetSimple(os.path.join('datasets', 'CUB'), split='train', transforms=train_transforms)
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=4, shuffle=True, num_workers=0)
    dataloader_train_iter = iter(dataloader_train)
    next(dataloader_train_iter)
    exit(0)