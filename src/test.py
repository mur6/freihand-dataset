import torch
from torch import nn
from torch.utils.data import DataLoader, dataloader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

batch_size = 1024

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

dataloader = train_dataloader

size = len(dataloader.dataset)
num_batches = len(dataloader)
print(f"sample_size={size} num_batches={num_batches}")
for iteration, (X, y) in enumerate(dataloader):
    print(f"iteration={iteration}")
    print(X.shape, y.shape, y[0].item())
print(f"sample_size={size} num_batches={num_batches}")

