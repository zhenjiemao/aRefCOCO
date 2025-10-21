from typing import Any
import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F_transforms
from PIL import Image
from torch.nn import functional as F


class Compose(object):
    def __init__(self, transforms) -> None:
        self.transforms = transforms
    def __call__(self, image , target) :

        for step in self.transforms:
            image, target = step(image, target)
        return image, target

class Resize(object):
    def __init__(self, output_size=224) -> None:
        self.size = output_size
    def __call__(self, image, target) -> Any:
        image = F_transforms.resize(image, self.size)

        target = F_transforms.resize(target, self.size, interpolation=T.InterpolationMode.NEAREST)
        
        return image, target

class Totensor(object):
    def __call__(self, image, target) :
        image = F_transforms.to_tensor(image)
        target = torch.tensor(np.asarray(target), dtype=torch.int64)
        return image, target

class Normalize(object):
    def __init__(self, mean, std) :
        self.mean = mean
        self.std = std
    
    def __call__(self, image , target):
        image = F_transforms.normalize(image, mean=self.mean, std=self.std)
        return image, target

def get_transform(size):
    transforms = []
        
    transforms.append(Resize(size))
    transforms.append(Totensor())
    transforms.append(Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))
    return Compose(transforms)


