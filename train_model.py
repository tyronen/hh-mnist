import logging
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

import utils
from models import Classifier

hyperparameters = {
    "batch_size": 128,
    "learning_rate": 0.001,
    "epochs": 5,
    "patch_size": 7,  # MNIST images are 28x28, so patch size of 7 -> 16 patches
    "model_dim": 64,
    "num_encoders": 6,
    "use_pe": True,  # whether to use positional encoding
}


def train(dataloader, model, loss_fn, optimizer, device, epoch):
    model.train()
    for batch, (X, y) in enumerate(tqdm(dataloader, f"Training epoch {epoch + 1}")):
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
        for X, y in tqdm(dataloader, "Testing"):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    logging.info(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    utils.setup_logging()
    logging.info("Downloading MNIST dataset...")

    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    training_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    stats_dataloader = DataLoader(training_data, batch_size=len(training_data.data), shuffle=False)
    images, _ = next(iter(stats_dataloader))
    mean = images.mean()
    std = images.std()

    train_dataloader = DataLoader(training_data, batch_size=hyperparameters["batch_size"], shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=hyperparameters["batch_size"], shuffle=False)

    device = utils.get_device()
    logging.info(f"Using {device} device")
    model = Classifier(
        patch_size=hyperparameters["patch_size"],
        model_dim=hyperparameters["model_dim"],
        num_encoders=hyperparameters["num_encoders"],
        use_pe=hyperparameters["use_pe"],
    )
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])

    for epoch in range(hyperparameters["epochs"]):
        train(train_dataloader, model, loss_fn, optimizer, device, epoch)

    test(test_dataloader, model, loss_fn, device)
    model_dict = {
        "model_state_dict": model.state_dict(),
        "mean": mean,
        "std": std,
    }
    torch.save(model_dict, utils.SIMPLE_MODEL_FILE)
    logging.info(f"Saved PyTorch Model State to {utils.SIMPLE_MODEL_FILE}")


if __name__ == "__main__":
    main()
