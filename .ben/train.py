import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from model import Classifier
from datetime import datetime
import utils as utils
from tqdm import tqdm
import wandb
import os

hyperparameters = {
    "batch_size": 1024,
    "learning_rate": 0.0005,
    "epochs": 100,
    "patch_size": 14,
    "dim_model": 64,
    "dropout_rate": 0.1,
    "num_encoders": 6,
    "has_positional_encoding": True,
    #noralization
    "has_input_norm": True,
    "has_pre_attention_norm": True,
    "has_post_attention_norm": True,
    "has_post_ffn_norm": True,
    "has_final_norm": False,
    # multi-head attention
    "num_attention_heads": 8
}

# normalization_layer_1: After patch projection
# normalization_layer_2: Before attention
# normalization_layer_3: Before feedforward (FFN) block
# normalization_layer_4: After adding positional encoding
# normalization_layer_5: Before final logits projection

def train(model, dataloader, loss_function, optimizer, device, epoch_num):
    model.train()
    running_loss = 0
    for batch_idx, (images, labels) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Epoch " + str(epoch_num) + " Training"):
        images = images.to(device)
        labels = labels.to(device)
        predictions = model(images)
        loss = loss_function(predictions, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()

    average_loss = running_loss / len(dataloader)
    return average_loss

def test(dataloader, model, device, epoch_num, loss_function):
    test_datasize = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, total=num_batches, desc="Epoch " + str(epoch_num) + " Testing"):
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images) # (batch_size, 10)
            test_loss += loss_function(predictions, labels).item() # (batch_size)
            correct += (predictions.argmax(1) == labels).type(torch.float).sum().item() # (batch_size)
    
    test_loss /= num_batches
    correct_rate = correct / test_datasize
    print(f"Epoch {epoch_num} Test Loss: {test_loss:.4f}, Test Accuracy: {100*correct_rate:.1f}, ({correct} out of {test_datasize})")
    return correct_rate, test_loss

def main():
    wandb.init(
        project="Transformer-MNIST", 
        config=hyperparameters,
        name=f"model-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    config = wandb.config

    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    training_data = datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(
        root="data", train=False, download=True, transform=transform
    )
    training_dataloader = DataLoader(training_data, batch_size=hyperparameters["batch_size"], shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=hyperparameters["batch_size"], shuffle=False)
    dim_k = config.dim_model // config.num_attention_heads
    dim_v = config.dim_model // config.num_attention_heads

    model = Classifier(
        patch_size=config.patch_size,
        dim_model=config.dim_model,
        dim_k=dim_k,
        dim_v=dim_v,
        dropout_rate=config.dropout_rate,
        has_positional_encoding=config.has_positional_encoding,
        has_input_norm=config.has_input_norm,
        has_post_attention_norm=config.has_post_attention_norm,
        has_post_ffn_norm=config.has_post_ffn_norm,
        has_pre_attention_norm=config.has_pre_attention_norm,
        has_final_norm=config.has_final_norm,
        num_encoders=config.num_encoders,
        num_heads=config.num_attention_heads
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    device = utils.get_device()
    model.to(device)
    
    best_score = 0                  # Initialize best_score
    best_model_state = None     # Store the best model state

    for epoch_num in range(config.epochs):
        train_loss = train(model, training_dataloader, loss_function, optimizer, device, epoch_num)
        correct_rate, test_loss = test(test_dataloader, model, device, epoch_num, loss_function)
        wandb.log({
            "test_accuracy": correct_rate,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "epoch": epoch_num
        })
        print({
            "test_accuracy": correct_rate,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "epoch": epoch_num
        })
        if correct_rate > best_score:
            best_score = correct_rate
            # Save the best model state (deep copy to avoid reference issues)
            best_model_state = {
                "model": model.state_dict().copy(),
                "optimizer": optimizer.state_dict().copy(),
                "epoch_num": epoch_num,
                "epochs": config.epochs,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "patch_size": config.patch_size,
                "patch_stride": config.patch_size,
                "dim_model": config.dim_model,
                "dropout_rate": config.dropout_rate,
                "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                "has_positional_encoding": config.has_positional_encoding,
                "has_input_norm": config.has_input_norm,
                "has_post_attention_norm": config.has_post_attention_norm,
                "has_post_ffn_norm": config.has_post_ffn_norm,
                "has_pre_attention_norm": config.has_pre_attention_norm,
                "has_final_norm": config.has_final_norm,
                "num_encoders": config.num_encoders,
                "num_attention_heads": config.num_attention_heads,
                "score": correct_rate,
            }
            print(f"New best score: {correct_rate:.4f}")

    # Save only the best model from this training session
    if best_model_state is not None:
        i = 1
        while os.path.exists(f"checkpoints/model-{i:04d}.pth"):
            i += 1
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(best_model_state, f"checkpoints/model-{i:04d}.pth")
        print(f"Best model saved to checkpoints/model-{i:04d}.pth with score {best_score:.4f}")
    else:
        print("No model saved - no improvement achieved")

    print("Training complete")

if __name__ == "__main__":
    main()