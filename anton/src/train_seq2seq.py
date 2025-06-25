import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
import editdistance

from inference import decode_sequence_greedy
from data import PAD_TOKEN


def compute_sequence_metrics(preds, labels):
    """
    preds: list of predicted token sequences
    labels: list of true token sequences
    Computes accuracy and average edit distance
    """
    total = len(preds)
    exact_match = 0
    edit_sum = 0

    for p, t in zip(preds, labels):
        # Remove padding
        p_clean = [x for x in p if x != PAD_TOKEN]
        t_clean = [x for x in t if x != PAD_TOKEN]

        if p_clean == t_clean:
            exact_match += 1
        edit_sum += editdistance.eval(p_clean, t_clean)

    acc = exact_match / total
    avg_edit = edit_sum / total
    return acc, avg_edit


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for x, y in tqdm(dataloader, desc="Training", leave=False):
        x, y = x.to(device), y.to(device)

        logits = model(x, y[:, :-1])  # input sequence to decoder (shifted)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def eval_one_epoch(model, dataloader, criterion, device, decode_method="greedy"):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Validating", leave=False):
            x, y = x.to(device), y.to(device)

            logits = model(x, y[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1))

            total_loss += loss.item()

            decoded_preds = decode_sequence_greedy(model, x)
            all_preds.extend(decoded_preds)
            all_labels.extend(y[:, 1:].tolist())

    avg_loss = total_loss / len(dataloader)
    acc, avg_edit = compute_sequence_metrics(all_preds, all_labels)
    return avg_loss, acc, avg_edit


def train_seq2seq(train_loader, val_loader, test_loader, device, model,
                num_epochs=10, lr=3e-4, decode_method="greedy", log_wandb=True):

    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if log_wandb:
        wandb.watch(model, log_freq=10)

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_edit = eval_one_epoch(model, val_loader, criterion, device,
                                                     decode_method=decode_method)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%, EditDist: {val_edit:.2f}")

        if log_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "val/edit_distance": val_edit
            })

    print("\nEvaluating on test set:")
    test_loss, test_acc, test_edit = eval_one_epoch(model, test_loader, criterion, device,
                                                    decode_method=decode_method)

    print(f"Test Loss: {test_loss:.4f}, Acc: {test_acc*100:.2f}%, EditDist: {test_edit:.2f}")

    if log_wandb:
        wandb.log({
            "test/loss": test_loss,
            "test/accuracy": test_acc,
            "test/edit_distance": test_edit
        })
