import torch
from PIL import Image
from torch.utils.data import Dataset

import torchvision.transforms as transforms
import os
import pickle
from PIL import ImageFile
from .cifar import get_cifar_10, get_cifar_100
# from .imagenet import get_imagenet
# from .tiny_imagenet import get_tiny_imagenet

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASET_DICT = {
    "cifar-10": get_cifar_10,
    "cifar-100": get_cifar_100,
    "cifar10": get_cifar_10,
    "cifar100": get_cifar_100,
    "CIFAR10": get_cifar_10,
    "CIFAR100": get_cifar_100,
    # "imagenet": get_imagenet,
    # "tiny-imagenet": get_tiny_imagenet
}


def get_dataset(name: str, root: str, split: str = "train", **kwargs) -> Dataset:
    fn = DATASET_DICT[name]
    return fn(root=root, split=split)


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def convert_binary(tensor):
    return tensor * (tensor==1).long()


# def convert_one_hot(target, num_classes=10):
#     a = torch.zeros([num_classes])
#     a[target] = 1
#     a = a.long()
#     return torch.LongTensor(a)
