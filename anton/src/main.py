import torch
import wandb
import random
import numpy as np
import argparse
from datetime import datetime
import os

from model import VisionTransformer
from data import load_mnist_dataloaders
from train import train_model
from utils import get_device
from config import load_config, override_config_with_wandb, extract_sweep_config


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main_internal(cfg):
    seed_all(cfg.train.seed)

    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = load_mnist_dataloaders(
        cfg.dataset.cache_dir,
        cfg.dataset.batch_size,
        cfg.dataset.valid_fraction,
        cfg.dataset.patch_size,
        cfg.train.seed)

    model = VisionTransformer(
        cfg.dataset.patch_size * cfg.dataset.patch_size,
        cfg.model.embed_dim,
        cfg.model.num_heads,
        cfg.model.mlp_dim,
        cfg.model.num_transformer_layers,
        cfg.dataset.num_classes,
        cfg.dataset.num_patches)

    train_model(
        train_loader,
        val_loader,
        test_loader,
        device,
        model,
        cfg.train.epochs,
        cfg.train.lr,
        log_wandb=cfg.log.wandb)

    # Save the trained model locally
    if "save_path_base" in cfg.model:
        run_id = wandb.run.id if cfg.log.wandb else "local"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = cfg.model.save_path_base
        dynamic_path = f"{base}_{timestamp}_{run_id}.pt"

        os.makedirs(os.path.dirname(dynamic_path), exist_ok=True)
        torch.save(model.state_dict(), dynamic_path)
        print(f"Model saved to {dynamic_path}")

        # Optionally upload to W&B
        if cfg.log.wandb:
            artifact = wandb.Artifact("trained-model", type="model")
            artifact.add_file(dynamic_path)
            wandb.log_artifact(artifact)


def main_with_wandb(base_cfg):
    if base_cfg.log.wandb:
        is_sweep = "WANDB_SWEEP_ID" in os.environ
        wandb.init(
            project=None if is_sweep else base_cfg.log.project,
            name=base_cfg.log.run_name,
            config=extract_sweep_config(base_cfg)
        )
        cfg = override_config_with_wandb(base_cfg, wandb.config)
    else:
        cfg = base_cfg

    main_internal(cfg)

    if cfg.log.wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main_with_wandb(cfg)
