# profile_model.py
import torch
import time
import utils
from models import ComplexTransformer
from train_complex_model import make_dataloader, hyperparameters

def run_profiler():
    """Run a diagnostic to isolate performance bottlenecks."""
    utils.setup_logging()
    device = utils.get_device()
    if device.type != 'cuda':
        print("This profiler requires a CUDA-enabled GPU to be meaningful.")
        return

    print(f"Running profiler on device: {device}")
    print("="*50)

    # --- 1. Profile the DataLoader ---
    print("Profiling DataLoader performance...")
    # Use your training dataloader settings
    train_dataloader = make_dataloader("data/composite_train.pt", device, shuffle=True)
    
    start_time = time.time()
    for i, batch in enumerate(train_dataloader):
        # We're only timing the data loading, so we do nothing with the batch
        if i == 0:
            # The first batch can be slow, so we start timing after it
            start_time = time.time()
    
    end_time = time.time()
    total_dataloader_time = end_time - start_time
    print(f"Time to iterate through the entire dataset: {total_dataloader_time:.2f} seconds.")
    print(f"This is the total time your GPU would be waiting for data per epoch.")
    print("="*50)

    # --- 2. Profile the Model's Computational Speed ---
    print("Profiling model forward/backward pass speed...")
    model = ComplexTransformer(
        patch_size=hyperparameters["patch_size"],
        model_dim=hyperparameters["model_dim"],
        ffn_dim=hyperparameters["ffn_dim"],
        num_coders=hyperparameters["num_coders"],
        num_heads=hyperparameters["num_heads"],
        dropout=hyperparameters["dropout"],
        train_pe=hyperparameters["train_pe"],
    ).to(device)

    model.optimizer = torch.optim.Adam(model.parameters())

    model.train()

    # Create a dummy batch of the correct size directly on the GPU
    batch_size = hyperparameters["batch_size"]
    dummy_images = torch.randn(batch_size, 1, 56, 56, device=device)
    dummy_input_seqs = torch.randint(0, 13, (batch_size, 5), device=device)
    
    # Use Automatic Mixed Precision (AMP) just like in training
    maybe_autocast, scaler = utils.amp_components(device, train=True)

    # Warm-up run (the first pass can be slow due to CUDA kernel compilation)
    with maybe_autocast:
        _ = model(dummy_images, dummy_input_seqs)

    # Timed run
    num_trials = 100
    torch.cuda.synchronize() # Wait for all previous CUDA work to finish
    start_time = time.time()
    
    for _ in range(num_trials):
        with maybe_autocast:
            logits = model(dummy_images, dummy_input_seqs)
            # A dummy loss is fine for profiling
            loss = logits.sum()
        
        scaler.scale(loss).backward()
        scaler.step(model.optimizer) # We need a dummy optimizer for this
        scaler.update()
        model.zero_grad(set_to_none=True)

    torch.cuda.synchronize() # Wait for the timed runs to finish
    end_time = time.time()
    
    avg_batch_time_ms = ((end_time - start_time) / num_trials) * 1000
    projected_epoch_time = (avg_batch_time_ms / 1000) * len(train_dataloader)
    
    print(f"Average time for one forward/backward pass: {avg_batch_time_ms:.2f} ms.")
    print(f"Projected epoch time based only on model computation: {projected_epoch_time:.2f} seconds.")
    print("="*50)


if __name__ == '__main__':
    run_profiler()