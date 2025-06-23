import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import logging

from tqdm import tqdm

import utils

hyperparameters = {
    "batch_size": 64,
    "learning_rate": 0.001,
    "epochs": 5,
}

def train(dataloader, model, loss_fn, optimizer, device):
    model.train()
    for batch, (X, y) in tqdm(enumerate(dataloader)):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    logging.info(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


def main():
    logging.info("Downloading MNIST dataset...")
    training_data = datasets.MNIST(
        root="data", train=True, download=True, transform=transforms.ToTensor
    )

    test_data = datasets.MNIST(
        root="data", train=False, download=True, transform=transforms.ToTensor
    )
    train_dataloader = DataLoader(training_data, batch_size=hyperparameters["batch_size"], shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=hyperparameters["batch_size"], shuffle=False)

    device = utils.get_device()
    logging.info(f"Using {device} device")
    model = nn.Module()
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])

    for t in range(hyperparameters["epochs"]):
        logging.info(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device)

    model_dict = {
        "model_state_dict": model.state_dict(),
    }
    torch.save(model_dict, utils.MODEL_FILE)
    logging.info(f"Saved PyTorch Model State to {utils.MODEL_FILE}")


if __name__ == "__main__":
    main()
