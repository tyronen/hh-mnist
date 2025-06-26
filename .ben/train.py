import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from model import Classifier
from datetime import datetime
import utils as utils
from tqdm import tqdm
import os

hyperparameters = {
    "batch_size": 1024,
    "learning_rate": 0.001,
    "epochs": 100,
    "patch_kernal_size": 14,
    "patch_stride": 14,
    "dim_model": 64,
    "dim_k": 64,
    "dim_v": 64,
    "has_positional_encoding": True
}

def train(model, dataloader, loss_function, optimizer, device, epoch_num):
    model.train()
    for batch_idx, (images, labels) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Epoch " + str(epoch_num) + " Training"):
        images = images.to(device)
        labels = labels.to(device)
        predictions = model(images)
        loss = loss_function(predictions, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test(dataloader, model, device, epoch_num, loss_function):
    test_datasize = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, total=num_batches, desc="Epoch " + str(epoch_num) + " Testing"):
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images) # (batch_size, 10)
            test_loss += loss_function(predictions, labels).item() # (batch_size)
            correct += (predictions.argmax(1) == labels).type(torch.float).sum().item() # (batch_size)
    
    test_loss /= num_batches
    correct_rate = correct / test_datasize
    print(f"Epoch {epoch_num} Test Loss: {test_loss:.4f}, Test Accuracy: {100*correct_rate:.1f}, ({correct} out of {test_datasize})")
    return correct_rate

def main():
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    training_data = datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(
        root="data", train=False, download=True, transform=transform
    )
    training_dataloader = DataLoader(training_data, batch_size=hyperparameters["batch_size"], shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=hyperparameters["batch_size"], shuffle=False)

    model = Classifier(
        patch_size=hyperparameters["patch_kernal_size"],
        stride=hyperparameters["patch_stride"],
        dim_model=hyperparameters["dim_model"],
        dim_k=hyperparameters["dim_k"],
        dim_v=hyperparameters["dim_v"],
        has_positional_encoding=hyperparameters["has_positional_encoding"]
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])
    device = utils.get_device()
    model.to(device)
    
    best_score = 0  # Initialize best_score

    for epoch_num in range(hyperparameters["epochs"]):
        train(model, training_dataloader, loss_function, optimizer, device, epoch_num)
        correct_rate = test(test_dataloader, model, device, epoch_num, loss_function)
        if correct_rate > best_score:
            best_score = correct_rate
            model_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epochs": hyperparameters["epochs"],
                "learning_rate": hyperparameters["learning_rate"],
                "batch_size": hyperparameters["batch_size"],
                "patch_kernal_size": hyperparameters["patch_kernal_size"],
                "patch_stride": hyperparameters["patch_stride"],
                "dim_model": hyperparameters["dim_model"],
                "dim_k": hyperparameters["dim_k"],
                "dim_v": hyperparameters["dim_v"],
                "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                "has_positional_encoding": hyperparameters["has_positional_encoding"],
                "score": correct_rate
            }
            i = 1
            while os.path.exists(f"checkpoints/model-{i:04d}.pth"):
                i += 1
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model_dict, f"checkpoints/model-{i:04d}.pth")
            print(f"Model saved to checkpoints/model-{i:04d}.pth with score {correct_rate:.2f}")

    print("Training complete")

if __name__ == "__main__":
    main()