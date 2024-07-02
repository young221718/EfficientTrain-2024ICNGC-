import torch
import torchvision
from torchvision.models import resnet18

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import numpy as np

from trainer import Trainer, Efficient_Trainer
from dataset import EffSampler, CustomCIFAR10

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transforms
transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


# Load CIFAR-10 dataset
trainset = CustomCIFAR10(root="./data", train=True, download=True, transform=transform)
eff_list = [float("inf") for _ in range(len(trainset))]
eff_list = np.array(eff_list)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=128,
    num_workers=2,
    sampler=EffSampler(trainset, eff_list),
)
print(len(trainloader))

testset = CustomCIFAR10(root="./data", train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2
)

# Define model
model = resnet18(pretrained=False, num_classes=10).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.004, momentum=0.9)

trainer = Efficient_Trainer(
    model=model,
    trainloader=trainloader,
    valloader=testloader,
    testloader=testloader,
    save_path="/home/prml/StudentsWork/ChanYoung/Experiments/classification/cifa10_resnet18/",
    criterion=criterion,
    optimizer=optimizer,
    scheduler=None,
    max_epoch=100,
    device=device,
    eff_list=eff_list,
)

trainer.train()
