import argparse
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets
from torchvision.transforms import v2
import logging
import wandb
from tqdm import tqdm
from contextlib import nullcontext

import utils
from models import Classifier

hyperparameters = {
    "batch_size": 128,
    "learning_rate": 0.0001,
    "epochs": 20,
    "patience": 2,
    "patch_size": 7,  # MNIST images are 28x28, so patch size of 7 -> 16 patches
    "model_dim": 64,
    "num_encoders": 3,
    "use_pe": True,  # whether to use positional encoding
    "seed": 42,
}

parser = argparse.ArgumentParser(description="Train simple model")
parser.add_argument("--entity", help="W and B entity", default="mlx-institute")
parser.add_argument("--project", help="W and B project", default="encoder-only")
args = parser.parse_args()


def amp_components(device, train=False):
    if device.type == "cuda" and train:
        return torch.cuda.amp.autocast, torch.cuda.amp.GradScaler()
    else:
        # fall-back: no automatic casting, dummy scaler
        return nullcontext, torch.cuda.amp.GradScaler(enabled=False)


def run_batch(
    dataloader,
    model,
    loss_fn,
    device,
    train: bool = False,
    optimizer=None,
    desc: str = "",
):
    """
    Runs one pass over `dataloader`.

    If `train` is True, the model is set to training mode and the optimizer is
    stepped. Otherwise the model is evaluated with torch.no_grad().
    Returns (accuracy %, average_loss) for the epoch.
    """
    model.train() if train else model.eval()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    total_loss, correct = 0.0, 0

    iterator = tqdm(dataloader, desc=desc)
    context = torch.enable_grad() if train else torch.no_grad()
    autocast, scaler = amp_components(device, train)
    with context:
        for X, y in iterator:
            X, y = X.to(device), y.to(device)
            with autocast():
                pred = model(X)
                loss = loss_fn(pred, y)

            total_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

    avg_loss = total_loss / num_batches
    accuracy = 100 * correct / size
    return accuracy, avg_loss


def main():
    utils.setup_logging()
    device = utils.get_device()
    logging.info(f"Using {device} device")

    run = wandb.init(entity=args.entity, project=args.project, config=hyperparameters)

    logging.info("Downloading MNIST dataset...")

    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    raw_data = datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )
    stats_dataloader = DataLoader(
        raw_data, batch_size=len(raw_data.data), shuffle=False
    )
    images, _ = next(iter(stats_dataloader))
    mean = images.mean()
    std = images.std()

    train_size = int(0.9 * len(raw_data))
    val_size = len(raw_data) - train_size
    generator = torch.Generator().manual_seed(hyperparameters["seed"])
    training_data, val_data = random_split(raw_data, [train_size, val_size], generator)
    test_data = datasets.MNIST(
        root="data", train=False, download=True, transform=transform
    )

    pin_memory = device.type == "cuda"
    num_workers = 4 if device.type == "cuda" else 0
    train_dataloader = DataLoader(
        training_data,
        batch_size=hyperparameters["batch_size"],
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=hyperparameters["batch_size"],
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=hyperparameters["batch_size"],
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )

    model = Classifier(
        patch_size=hyperparameters["patch_size"],
        model_dim=hyperparameters["model_dim"],
        num_encoders=hyperparameters["num_encoders"],
        use_pe=hyperparameters["use_pe"],
    )
    model.to(device)
    wandb.watch(model, log="all", log_freq=100)
    wandb.define_metric("val_accuracy", summary="max")
    wandb.define_metric("val_loss", summary="min")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])

    best_loss = float("inf")
    epochs_since_best = 0
    for epoch in range(hyperparameters["epochs"]):
        train_correct, train_loss = run_batch(
            train_dataloader,
            model,
            loss_fn,
            device,
            train=True,
            optimizer=optimizer,
            desc=f"Training epoch {epoch + 1}",
        )
        val_correct, val_loss = run_batch(
            val_dataloader,
            model,
            loss_fn,
            device,
            train=False,
            desc=f"Validating epoch {epoch + 1}",
        )
        run.log(
            {
                "train_accuracy": train_correct,
                "train_loss": train_loss,
                "val_accuracy": val_correct,
                "val_loss": val_loss,
            },
        )
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_since_best = 0
            model_dict = {
                "model_state_dict": model.state_dict(),
                "mean": mean,
                "std": std,
            }
            torch.save(model_dict, utils.SIMPLE_MODEL_FILE)
        else:
            epochs_since_best += 1
        if epochs_since_best >= hyperparameters["patience"]:
            break

    checkpoint = torch.load(utils.SIMPLE_MODEL_FILE)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_correct, test_loss = run_batch(
        test_dataloader, model, loss_fn, device, train=False, desc="Testing"
    )
    run.log({"test_accuracy": test_correct, "test_loss": test_loss})
    logging.info(f"Saved PyTorch Model State to {utils.SIMPLE_MODEL_FILE}")
    run.finish(0)


if __name__ == "__main__":
    main()
