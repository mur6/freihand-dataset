import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_rgb = transforms.Lambda(lambda image: image.convert('RGB'))
my_transform = transforms.Compose([
    to_rgb,
    ToTensor(),
    normalize])


def get_dataloaders():
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=my_transform
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=my_transform
    )
    batch_size = 128
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    return train_dataloader, test_dataloader


def get_pretrained_model():
    model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
    #model = torchvision.models.vgg16(pretrained=True)
    #print(model)
    # for param in model.parameters():
    #     param.requires_grad = False
    model.fc = torch.nn.Linear(512, 10)
    print(model)
    return model

def get_pretrained_model_1():
    #model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
    model = torchvision.models.vgg16(pretrained=True)
    #print(model)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[-1] = torch.nn.Linear(4096, 10)
    print(model)
    return model

def get_pretrained_model_2():
    model = torchvision.models.mobilenet.mobilenet_v3_small(pretrained=True)
    print(model)
    for param in model.parameters():
        param.requires_grad = False
    #model.classifier[-1] = torch.nn.Linear(1280, 10)
    print(model)
    return model

get_pretrained_model()
