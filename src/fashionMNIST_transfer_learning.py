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
    transforms.Resize(224),
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
    batch_size = 256
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader

def get_pretrained_model():
    model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
    #model = torchvision.models.vgg16(pretrained=True)
    model.fc = torch.nn.Linear(512, 10)
    return model

def get_pretrained_model_vgg():
    model = torchvision.models.vgg11_bn(pretrained=True)
    #model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
    #for param in model.parameters():
    #    param.requires_grad = False
    #first_conv_layer = [nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=True)]
    #first_conv_layer.extend(list(model.features))
    #model.features= nn.Sequential(*first_conv_layer)
    model.classifier[-1] = torch.nn.Linear(4096, 10)
    return model

def get_pretrained_model_mobilenet():
    model = torchvision.models.mobilenet.mobilenet_v2(pretrained=True)
    #print(model)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[-1] = torch.nn.Linear(1280, 10)
    return model

def train_loop(dataloader, device, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    print(f"sample_size={size} num_batches={num_batches}")


def test_loop(dataloader, device, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataloader, test_dataloader = get_dataloaders()
    model = get_pretrained_model()
    model = model.to(device)

    learning_rate = 0.0005
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, device, model, criterion, optimizer)
        test_loop(test_dataloader, device, model, criterion)

main()
print("Done!")
