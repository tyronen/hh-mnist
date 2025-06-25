import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import logging

from tqdm import tqdm

import utils
from create_composite_images import PAD_TOKEN
from models import ComplexTransformer

hyperparameters = {
    "batch_size": 1024,
    "learning_rate": 0.001,
    "epochs": 5,
    "patience": 2,
    "patch_size": 14,
    "model_dim": 64,
    "text_dim": 11,
    "ffn_dim": 64,
    "num_coders": 3,
    "num_heads": 8,
    "seed": 42,
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
        logits = model(images, input_seqs)  # (B, seq_len, VOCAB)
        loss = loss_fn(
            logits.view(-1, logits.size(-1)), output_seqs.view(-1)  # (B*seq_len, VOCAB)
        )  # (B*seq_len)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def validate(dataloader, model, loss_fn, device, epoch):
    size = 0
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for images, input_seqs, output_seqs in tqdm(
            dataloader, f"Validating epoch {epoch}"
        ):
            logits = model(images, input_seqs)
            test_loss += loss_fn(logits.view(-1, logits.size(-1)), output_seqs.view(-1))
            mask = output_seqs != PAD_TOKEN
            correct += ((logits.argmax(-1) == output_seqs) & mask).sum().item()
            size += mask.sum().item()
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
    model = ComplexTransformer(
        patch_size=hyperparameters["patch_size"],
        model_dim=hyperparameters["model_dim"],
        ffn_dim=hyperparameters["ffn_dim"],
        num_coders=hyperparameters["num_coders"],
        num_heads=hyperparameters["num_heads"],
        text_dim=hyperparameters["text_dim"],
    ).to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
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
