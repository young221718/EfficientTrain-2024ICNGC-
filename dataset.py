import torch
from torchvision.datasets import CIFAR10


class EffSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, weights):
        self.dataset = dataset
        self.weights = weights

    def __iter__(self):
        indices = sorted(
            range(len(self.dataset)), key=lambda i: self.weights[i], reverse=True
        )
        for i in indices:
            yield i

    def __len__(self):
        return len(self.dataset)


class CustomCIFAR10(CIFAR10):
    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False
    ):
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

    def __getitem__(self, index):
        image, target = super().__getitem__(index)

        return index, image, target
