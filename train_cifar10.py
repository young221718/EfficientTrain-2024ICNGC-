import torch
import torchvision
from torchvision.models import resnet18

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from trainer import Trainer
from dataset import CustomCIFAR10

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
trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2
)

# Define model
model = resnet18(pretrained=False, num_classes=10).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.004, momentum=0.9)

trainer = Trainer(
    model=model,
    trainloader=trainloader,
    valloader=testloader,
    testloader=testloader,
    save_path="/home/prml/StudentsWork/ChanYoung/Experiments/classification/cifa10_resnet18/",
    criterion=criterion,
    optimizer=optimizer,
    scheduler=None,
    max_epoch=50,
    device=device,
)

trainer.train()
