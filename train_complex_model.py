import argparse
from typing import Optional

import torch
from torch import nn, optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset
import logging

from tqdm import tqdm

import utils
from create_composite_images import PAD_TOKEN, END_TOKEN
from models import ComplexTransformer
import wandb

hyperparameters = {
    "batch_size": 256,
    "learning_rate": 0.001,
    "epochs": 20,
    "patience": 2,
    "patch_size": 14,
    "model_dim": 256,
    "ffn_dim": 1024,
    "num_coders": 6,
    "num_heads": 8,
    "seed": 42,
}

parser = argparse.ArgumentParser(description="Train simple model")
parser.add_argument("--entity", help="W and B entity", default="mlx-institute")
parser.add_argument("--project", help="W and B project", default="encoder-decoder")
args = parser.parse_args()


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


def run_batch(
    dataloader,
    model,
    device,
    loss_fn,
    train: bool = False,
    optimizer: Optional[Optimizer] = None,
    desc: str = "",
):
    model.train() if train else model.eval()

    total_tokens = 0
    num_batches = len(dataloader)
    total_loss, num_correct_digits = 0.0, 0
    seq_total, seq_correct = 0, 0
    iterator = tqdm(dataloader, desc=desc)
    context = torch.enable_grad() if train else torch.no_grad()
    maybe_autocast, scaler = utils.amp_components(device, train)
    with context:
        for images, input_seqs, output_seqs in iterator:
            images, input_seqs, output_seqs = (
                images.to(device),
                input_seqs.to(device),
                output_seqs.to(device),
            )
            with maybe_autocast:
                logits = model(images, input_seqs)
                loss = loss_fn(
                    logits.view(-1, logits.size(-1)),
                    output_seqs.view(-1),  # (B*seq_len, VOCAB)
                )  # (B*seq_len)

                total_loss += loss
                mask = (output_seqs != PAD_TOKEN) & (output_seqs != END_TOKEN)
                pred = logits.argmax(-1)
                batch_num_correct_digits = ((pred == output_seqs) & mask).sum().item()
                num_correct_digits += batch_num_correct_digits
                batch_tokens = mask.sum().item()
                total_tokens += batch_tokens
                batch_seq_correct = (
                    ((pred == output_seqs) | ~mask).all(dim=1).sum().item()
                )
                batch_seq_size = output_seqs.size(0)
                seq_correct += batch_seq_correct
                seq_total += batch_seq_size
                wandb.log(
                    {
                        "batch_loss": loss.item(),
                        "batch_non_padded_tokens": batch_tokens,
                        "batch_digits_correct": batch_num_correct_digits,
                        "batch_seq_correct": batch_seq_correct,
                        "batch_seq_size": batch_seq_size,
                    }
                )

            if train:
                if optimizer is None:
                    raise ValueError("Optimizer must be provided when train=True")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

    avg_loss = total_loss / num_batches
    wandb.log(
        {
            "total_loss": total_loss,
            "num_batches": num_batches,
            "digits_correct": num_correct_digits,
            "total_tokens": total_tokens,
            "seq_total": seq_total,
            "seq_correct": seq_correct,
        }
    )
    token_accuracy = 100 * num_correct_digits / total_tokens
    seq_accuracy = 100 * seq_correct / seq_total
    return token_accuracy, seq_accuracy, avg_loss


# true and true = true
# true and false = false
# false and false = false

# true or not-true = true
# true or not-true = false
# false or not-false = true


def run_single_training(config=None):
    """Run a single training session with given config."""
    if config is None:
        config = hyperparameters

    device = utils.get_device()
    logging.info(f"Using {device} device.")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if device.type == "cuda":
        torch.set_float32_matmul_precision("medium")
    train_dataloader = make_dataloader("data/composite_train.pt", device, shuffle=True)
    val_dataloader = make_dataloader("data/composite_val.pt", device, shuffle=False)
    test_dataloader = make_dataloader("data/composite_test.pt", device, shuffle=False)
    model = ComplexTransformer(
        patch_size=hyperparameters["patch_size"],
        model_dim=hyperparameters["model_dim"],
        ffn_dim=hyperparameters["ffn_dim"],
        num_coders=hyperparameters["num_coders"],
        num_heads=hyperparameters["num_heads"],
    ).to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])

    model = run_training(
        model=model,
        train_dl=train_dataloader,
        val_dl=val_dataloader,
        device=device,
        loss_fn=loss_fn,
        optimizer=optimizer,
        config=config,
    )

    # if we stopped early and have a checkpoint, load it
    checkpoint = torch.load(utils.COMPLEX_MODEL_FILE)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_token_accuracy, test_seq_accuracy, test_loss = run_batch(
        dataloader=test_dataloader,
        model=model,
        device=device,
        loss_fn=loss_fn,
        train=False,
        desc="Testing",
    )
    wandb.log(
        {
            "test_token_accuracy": test_token_accuracy,
            "test_seq_accuracy": test_seq_accuracy,
            "test_loss": test_loss,
        }
    )

    return model


def main():
    utils.setup_logging()
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        config=hyperparameters,
    )

    run_single_training(hyperparameters)
    artifact = wandb.Artifact(name="complex_model", type="model")
    artifact.add_file(utils.COMPLEX_MODEL_FILE)
    run.log_artifact(artifact)
    run.finish(0)


def run_training(
    model: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    config: dict,
) -> nn.Module:
    wandb.watch(model, log="all", log_freq=100)
    wandb.define_metric("val_token_accuracy", summary="max")
    wandb.define_metric("val_seq_accuracy", summary="max")
    wandb.define_metric("val_loss", summary="min")
    best_loss = float("inf")
    epochs_since_best = 0
    for epoch in range(config["epochs"]):
        train_token_accuracy, train_seq_accuracy, train_loss = run_batch(
            dataloader=train_dl,
            model=model,
            device=device,
            train=True,
            loss_fn=loss_fn,
            optimizer=optimizer,
            desc=f"Training epoch {epoch + 1}",
        )
        val_token_accuracy, val_seq_accuracy, val_loss = run_batch(
            dataloader=val_dl,
            model=model,
            device=device,
            train=False,
            loss_fn=loss_fn,
            desc=f"Validating epoch {epoch + 1}",
        )
        wandb.log(
            {
                "train_token_accuracy": train_token_accuracy,
                "train_seq_accuracy": train_seq_accuracy,
                "train_loss": train_loss,
                "val_token_accuracy": val_token_accuracy,
                "val_seq_accuracy": val_seq_accuracy,
                "val_loss": val_loss,
            },
        )
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_since_best = 0
            model_dict = {
                "model_state_dict": model.state_dict(),
                "config": config,
            }
            torch.save(model_dict, utils.COMPLEX_MODEL_FILE)
        else:
            epochs_since_best += 1
        if epochs_since_best >= config["patience"] or best_loss == 0:
            break

    return model


if __name__ == "__main__":
    main()
