import click
import torchvision.datasets.mnist as mnist
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
import torch.nn as nn
from model import CapsuleNetwork
from tqdm import tqdm
import torch.optim as optim
import torch


@click.command()
@click.option('--epochs', default=5)
def main(epochs):
    if torch.cuda.is_available:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    trainset = mnist.MNIST(root="~/pytorch", train=True, download=True, transform=transforms.ToTensor())
    testset = mnist.MNIST(root="~/pytorch", train=False, download=True)
    trainloader = DataLoader(trainset, 32, shuffle=True)
    testloader = DataLoader(testset, 32)
    capsnet = CapsuleNetwork().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(capsnet.parameters(), lr=0.001)

    for epoch in range(epochs):
        pbar = tqdm(trainloader)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = capsnet(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            pbar.set_description("Loss: {:0.4f}".format(loss.item()))


if __name__ == "__main__":
    main()
