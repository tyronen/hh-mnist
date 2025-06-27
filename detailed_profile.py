# detailed_profile.py
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import utils
from models import ComplexTransformer
from train_complex_model import hyperparameters

def run_detailed_profiler():
    """Uses the PyTorch Profiler to find operator-level bottlenecks."""
    utils.setup_logging()
    device = utils.get_device()
    if device.type != 'cuda':
        print("This profiler requires a CUDA-enabled GPU.")
        return

    print(f"Running detailed profiler on device: {device}")
    print("="*50)

    # 1. Set up the model and a single batch of dummy data on the GPU
    model = ComplexTransformer(
        patch_size=hyperparameters["patch_size"],
        model_dim=hyperparameters["model_dim"],
        ffn_dim=hyperparameters["ffn_dim"],
        num_coders=hyperparameters["num_coders"],
        num_heads=hyperparameters["num_heads"],
        dropout=hyperparameters["dropout"],
        train_pe=hyperparameters["train_pe"],
    ).to(device)
    model.train()
    
    # We use a dummy optimizer for profiling purposes
    optimizer = torch.optim.Adam(model.parameters())
    
    dummy_images = torch.randn(hyperparameters["batch_size"], 1, 56, 56, device=device)
    dummy_input_seqs = torch.randint(0, 13, (hyperparameters["batch_size"], 5), device=device)
    
    # We will trace 5 steps: 1 wait, 1 warmup, 3 active steps
    profiler_schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)

    # 2. Run the profiler
    # The on_trace_ready handler will export a trace file for TensorBoard
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=profiler_schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/profile'),
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for i in range(5):
            with record_function(f"step_{i}"):
                logits = model(dummy_images, dummy_input_seqs)
                loss = logits.sum() # A simple dummy loss is all we need
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            # The profiler needs to be stepped at the end of each iteration
            prof.step()

    # 3. Print the summary report
    print("Profiler run complete. Printing summary tables...")
    print("="*50)
    
    # Print a summary table sorted by total CUDA time
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    print("\nTo see the detailed trace, run the following command in your terminal:")
    print("tensorboard --logdir=./logs")


if __name__ == '__main__':
    run_detailed_profiler()