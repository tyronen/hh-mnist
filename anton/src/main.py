import torch
import wandb
import argparse
from datetime import datetime
import os

from model import ViTClassifier, VisionToSequence
from data import load_mnist_dataloaders
from train_classifier import train_classifier
from train_seq2seq import train_seq2seq
from utils import get_device, seed_all
from config import load_config, override_config_with_wandb, extract_sweep_config


def main_internal(cfg, mode):
    seed_all(cfg.train.seed)

    device = get_device()
    print(f"Using device: {device}")

    if mode == "classifier":
        train_loader, val_loader, test_loader = load_mnist_dataloaders(
            cfg.dataset.cache_dir,
            cfg.dataset.batch_size,
            cfg.dataset.valid_fraction,
            cfg.dataset.patch_size,
            cfg.train.seed)

        model = ViTClassifier(
            cfg.dataset.patch_size * cfg.dataset.patch_size,
            cfg.model.embed_dim,
            cfg.model.num_heads,
            cfg.model.mlp_dim,
            cfg.model.num_transformer_layers,
            cfg.dataset.num_classes,
            cfg.dataset.num_patches,
            cfg.model.avg_pooling,
            cfg.model.add_pos_emb,
            cfg.model.dropout,
            cfg.model.dot_product_norm)

        train_classifier(
            train_loader,
            val_loader,
            test_loader,
            device,
            model,
            cfg.train.epochs,
            cfg.train.lr,
            log_wandb=cfg.log.wandb)

    elif mode == "seq2seq":
        train_loader, val_loader, test_loader = load_mnist_dataloaders(
            cfg.dataset.cache_dir,
            cfg.dataset.batch_size,
            cfg.dataset.valid_fraction,
            cfg.dataset.patch_size,
            cfg.train.seed,
            composite_mode=True,
            canvas_size=(cfg.dataset.canvas_size_w, cfg.dataset.canvas_size_h),
            num_digits=cfg.dataset.num_digits,
            placement=cfg.dataset.placement,
            num_digits_range=None, # TODO Support this cfg.dataset.num_digits_range,
            num_images=cfg.dataset.num_images,
            num_images_test=cfg.dataset.num_images_test)
        
        model = VisionToSequence(
            cfg.dataset.patch_size * cfg.dataset.patch_size,
            max_len=10,  # FIXME auto-detect this number
            embed_dim=cfg.model.embed_dim,
            num_heads=cfg.model.num_heads,
            mlp_dim=cfg.model.mlp_dim,
            num_layers_encoder=cfg.model.num_layers_encoder,
            num_layers_decoder=cfg.model.num_layers_decoder,
            num_patches=cfg.dataset.num_patches)

            # TODO Expose these parameters
            # cfg.model.add_pos_emb,
            # cfg.model.dropout,
            # cfg.model.dot_product_norm

        train_seq2seq(
            train_loader,
            val_loader,
            test_loader,
            device,
            model,
            cfg.train.epochs,
            cfg.train.lr,
            log_wandb=cfg.log.wandb)

    else:
        raise ValueError(f"Invalid mode: '{mode}'. Mode should be one of 'classifier' or 'seq2seq'.")
        
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


def main_with_wandb(base_cfg, mode):
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

    main_internal(cfg, mode)

    if cfg.log.wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=["classifier", "seq2seq"],
        help="Select the mode: 'classifier' or 'seq2seq'"
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    main_with_wandb(cfg, args.mode)
