import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import wandb


def evaluate_classification_metrics(preds, labels, prefix="", log_wandb=False):
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')

    # Log to stdout
    print(f"{prefix}Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # Log to wandb
    if log_wandb:
        wandb.log({
            f"{prefix}accuracy": acc,
            f"{prefix}precision": precision,
            f"{prefix}recall": recall,
            f"{prefix}f1": f1
        })

        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{prefix}Confusion Matrix")

        wandb.log({f"{prefix}confusion_matrix": wandb.Image(fig)})
        plt.close(fig)


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for x, y in tqdm(dataloader, desc="Training", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(y.cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def eval_one_epoch(model, dataloader, criterion, device, return_preds_labels=False):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Validating", leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item()
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)

    if return_preds_labels:
        return avg_loss, acc, all_preds, all_labels
    return avg_loss, acc


def train_model(train_loader, val_loader, test_loader, device, model, num_epochs = 10, lr=3e-4, log_wandb=False):
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if log_wandb:
        wandb.watch(model, log_freq=10)

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc*100:.2f}%")

        if log_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "val/loss": val_loss,
                "val/accuracy": val_acc
            })

    # Final test evaluation
    print("\nEvaluating on test set:")
    test_loss, test_acc, test_preds, test_labels = eval_one_epoch(
        model, test_loader, criterion, device, return_preds_labels=True
    )
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc*100:.2f}%")

    # Log classification metrics and confusion matrix
    evaluate_classification_metrics(test_preds, test_labels, prefix="test/", log_wandb=log_wandb)
