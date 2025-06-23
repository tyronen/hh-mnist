import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import logging

from tqdm import tqdm

import utils
from models import Predictor

hyperparameters = {
    "batch_size": 64,
    "learning_rate": 0.001,
    "epochs": 5,
}


def make_dataloader(path, device, shuffle):
    tensors = torch.load(path, map_location=device)
    dataset = TensorDataset(
        tensors["images"].to(device),
        tensors["input_seqs"].to(device),
        tensors["output_seqs"].to(device),
    )
    return DataLoader(
        dataset, batch_size=hyperparameters["batch_size"], shuffle=shuffle
    )


def train(dataloader, model, loss_fn, optimizer, device, epoch):
    model.train()
    for images, input_seqs, output_seqs in tqdm(dataloader, f"Training epoch {epoch}"):
        pred = model(images, input_seqs)
        loss = loss_fn(pred, output_seqs)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def validate(dataloader, model, loss_fn, device, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for images, input_seqs, output_seqs in tqdm(
            dataloader, f"Validating epoch {epoch}"
        ):
            pred = model(images, input_seqs)
            test_loss += loss_fn(pred, output_seqs)
            correct += (pred.argmax(1) == output_seqs).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    logging.info(
        f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


def main():
    utils.setup_logging()
    device = utils.get_device()
    train_dataloader = make_dataloader("data/composite_train.pt", device, shuffle=True)
    val_dataloader = make_dataloader("data/composite_val.pt", device, shuffle=False)
    test_dataloader = make_dataloader("data/composite_test.pt", device, shuffle=False)
    model = Predictor().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])

    for epoch in range(hyperparameters["epochs"]):
        train(train_dataloader, model, loss_fn, optimizer, device, epoch + 1)
        validate(val_dataloader, model, loss_fn, device, epoch + 1)

    validate(test_dataloader, model, loss_fn, device, "Test")
    model_dict = {
        "model_state_dict": model.state_dict(),
    }
    torch.save(model_dict, utils.COMPLEX_MODEL_FILE)
    logging.info(f"Saved PyTorch Model State to {utils.COMPLEX_MODEL_FILE}")


if __name__ == "__main__":
    main()
