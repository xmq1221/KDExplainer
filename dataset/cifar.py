import torch
from torchvision import transforms
from torchvision.datasets import CIFAR100, CIFAR10
from torch.utils.data import Dataset


def convert_one_hot_cifar10(target, num_classes=10):
    a = torch.zeros([num_classes])
    a[target] = 1
    a = a.long()
    return torch.LongTensor(a)


def convert_one_hot_cifar100(target, num_classes=100):
    a = torch.zeros([num_classes])
    a[target] = 1
    a = a.long()
    return torch.LongTensor(a)


def get_cifar_10(root: str, split: str = "train") -> Dataset:
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    if split == "train":
        transform = train_transform
        is_train = True
    else:
        transform = test_transform
        is_train = False
    
    # if loss_method == 'nce':
    #     target_transform = convert_one_hot_cifar10
    # elif loss_method == 'ce':
    #     target_transform = None

    target_transform = None

    dataset = CIFAR10(
        root=root,
        train=is_train,
        transform=transform,
        target_transform=target_transform,
        download=True
    )

    return dataset


def get_cifar_100(root: str,  split: str = "train") -> Dataset:
    normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    if split == "train":
        transform = train_transform
        is_train = True
    else:
        transform = test_transform
        is_train = False

    # if loss_method == 'nce':
    #     target_transform = convert_one_hot_cifar100
    # elif loss_method == 'ce':
    #     target_transform = None

    target_transform = None

    dataset = CIFAR100(
        root=root,
        train=is_train,
        transform=transform,
        target_transform=target_transform,
        download=True
    )

    return dataset
