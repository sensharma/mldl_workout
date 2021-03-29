import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


def get_split_indeces(n_samples, val_split):
    indices = list(range(n_samples))
    split = int(np.floor(val_split * n_samples))
    train_indices, val_indices = indices[split:], indices[:split]
    return train_indices, val_indices


class TorchVisionDataset:
    def __init__(self, dataset_name):
        if not hasattr(torchvision.datasets, dataset_name):
            raise ValueError(f"Dataset {dataset_name} is not in torchvision.datasets")
        self.dataset_name = dataset_name
        self.torchvision_loader = getattr(torchvision.datasets, dataset_name)
        self.training_set = None
        self.test_set = None

    def init_train_dataset(self, **kwargs):
        self.training_set = self.torchvision_loader(train=True, **kwargs)

    def get_train_dataloader(self, batch_size=16, val_split=None, **kwargs):
        assert self.training_set is not None, "Call init_train_dataset first"
        validation_loader = None
        if val_split:
            train_indices, val_indices = get_split_indeces(len(self.training_set), val_split)
            train_sampler = SubsetRandomSampler(train_indices)
            valid_sampler = SubsetRandomSampler(val_indices)
            train_loader = torch.utils.data.DataLoader(
                self.training_set,
                batch_size=batch_size,
                sampler=train_sampler,
                **kwargs,
            )
            validation_loader = torch.utils.data.DataLoader(
                self.training_set,
                batch_size=batch_size,
                sampler=valid_sampler,
                **kwargs,
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                self.training_set,
                batch_size=batch_size,
                **kwargs,
            )
        return train_loader, validation_loader

    def init_test_dataset(self, **kwargs):
        self.test_set = self.torchvision_loader(train=False, **kwargs)

    def get_test_dataloader(self, batch_size=16, **kwargs):
        assert self.test_set is not None, "Call init_test_dataset first"
        test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=batch_size, **kwargs)
        return test_loader


def get_loaders_cifar10(batch_size=4):

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    classes = trainset.classes

    return trainloader, testloader, classes
