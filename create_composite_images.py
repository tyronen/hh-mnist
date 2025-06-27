import torch
import numpy as np
from torchvision import datasets
from torchvision.transforms import v2
import random
import os
import utils
from tqdm import tqdm
import logging

from utils import START_TOKEN, END_TOKEN, BLANK_TOKEN


# Define special tokens


def create_composite_image(mnist_images, mnist_labels, num_images):
    """Create a 56x56 composite image with num_images MNIST digits"""
    composite = torch.zeros(1, 56, 56)
    labels = []

    # Define the 4 quadrant positions - top left, top right, bottom left, bottom right
    positions = [(0, 0), (0, 28), (28, 0), (28, 28)]
    filled = [False, False, False, False]

    # Randomly select which positions to fill
    positions_to_fill = random.sample(range(4), num_images)
    for pos in positions_to_fill:
        filled[pos] = True

    # Randomly select MNIST images
    indices = random.choices(range(len(mnist_images)), k=num_images)
    last_index = 0

    for pos, is_filled in zip(positions, filled):
        if not is_filled:
            labels.append(BLANK_TOKEN)
            continue
        # Place the 28x28 MNIST image at the selected position
        y, x = pos
        idx = indices[last_index]
        composite[0, y : y + 28, x : x + 28] = mnist_images[idx][0]
        labels.append(mnist_labels[idx])
        last_index += 1

    return composite, labels


def create_transformer_seqs(labels):
    """Create input and output sequences for transformer training"""
    # Input: START_TOKEN + labels (padded to max length)
    # Output: labels + END_TOKEN (padded to max length)

    assert len(labels) == 4

    # Input sequence: [START_TOKEN, label1, label2]
    # We don't truncate the labels, because this is now a fixed length
    input_seq = [START_TOKEN] + labels

    # Output sequence: [label1, label2, ..., END_TOKEN]
    output_seq = labels + [END_TOKEN]

    return torch.tensor(input_seq), torch.tensor(output_seq)


def generate_composite_dataset(mnist_dataset, size, distribution):
    """Generate composite dataset with given size and distribution"""
    composite_images = []
    input_seqs = []
    output_seqs = []

    # Convert MNIST data to lists for easier random access
    mnist_images = []
    mnist_labels = []
    for img, label in tqdm(mnist_dataset, desc="Transforming MNIST"):
        mnist_images.append(img)
        mnist_labels.append(label)

    for i in tqdm(range(size), desc="Generating composites"):
        # Determine number of images based on distribution
        rand = random.random()
        if rand < 0.4:  # 40% - 4 images
            num_images = 4
        elif rand < 0.7:  # 30% - 3 images
            num_images = 3
        elif rand < 0.9:  # 20% - 2 images
            num_images = 2
        else:  # 10% - 1 image
            num_images = 1

        composite_img, labels = create_composite_image(
            mnist_images, mnist_labels, num_images
        )
        input_seq, output_seq = create_transformer_seqs(labels)
        composite_images.append(composite_img)
        input_seqs.append(input_seq)
        output_seqs.append(output_seq)

    return (
        torch.stack(composite_images),
        torch.stack(input_seqs),
        torch.stack(output_seqs),
    )


def split_dataset(images, input_seqs, output_seqs, val_split=0.2):
    """Split dataset into training and validation sets"""
    dataset_size = len(images)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size

    # Create random indices
    indices = torch.randperm(dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_images = images[train_indices]
    train_input_seqs = input_seqs[train_indices]
    train_output_seqs = output_seqs[train_indices]

    val_images = images[val_indices]
    val_input_seqs = input_seqs[val_indices]
    val_output_seqs = output_seqs[val_indices]

    return (
        train_images,
        train_input_seqs,
        train_output_seqs,
        val_images,
        val_input_seqs,
        val_output_seqs,
    )


def save_torch_dataset(images, input_seqs, output_seqs, filepath):
    """Save dataset as torch tensors"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    torch.save(
        {
            "images": images,
            "input_seqs": input_seqs,
            "output_seqs": output_seqs,
            "vocab_info": {
                "start_token": START_TOKEN,
                "end_token": END_TOKEN,
                "blank_token": BLANK_TOKEN,
                "vocab_size": 13,  # 0-9 digits + START + END + PAD
            },
        },
        filepath,
    )

    logging.info(f"Saved dataset to {filepath}")


def main():
    utils.setup_logging()
    logging.info("Downloading original MNIST dataset...")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Load original MNIST data
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            # add mouse-like behaviour
            v2.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        ]
    )

    training_data = datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(
        root="data", train=False, download=True, transform=transform
    )

    logging.info(f"Original training set size: {len(training_data)}")
    logging.info(f"Original test set size: {len(test_data)}")

    # Generate composite datasets
    original_train_size = len(training_data)  # 60,000
    test_size = len(test_data)  # 10,000

    # Distribution: 40% have 4 images, 30% have 3, 20% have 2, 10% have 1
    distribution = [0.4, 0.3, 0.2, 0.1]

    logging.info("Generating composite training dataset...")
    all_train_images, all_train_input_seqs, all_train_output_seqs = (
        generate_composite_dataset(training_data, original_train_size, distribution)
    )

    logging.info("Splitting training set into train/validation (80%/20%)...")
    (
        train_images,
        train_input_seqs,
        train_output_seqs,
        val_images,
        val_input_seqs,
        val_output_seqs,
    ) = split_dataset(
        all_train_images, all_train_input_seqs, all_train_output_seqs, val_split=0.2
    )

    logging.info("Generating composite test dataset...")
    test_images, test_input_seqs, test_output_seqs = generate_composite_dataset(
        test_data, test_size, distribution
    )

    # Save datasets
    logging.info("Saving datasets...")
    save_torch_dataset(
        train_images, train_input_seqs, train_output_seqs, "data/composite_train.pt"
    )
    save_torch_dataset(
        val_images, val_input_seqs, val_output_seqs, "data/composite_val.pt"
    )
    save_torch_dataset(
        test_images, test_input_seqs, test_output_seqs, "data/composite_test.pt"
    )

    # logging.info statistics
    logging.info("Dataset Statistics:")
    logging.info(
        f"Training set: {len(train_images)} images of size {train_images[0].shape}"
    )
    logging.info(
        f"Validation set: {len(val_images)} images of size {val_images[0].shape}"
    )
    logging.info(f"Test set: {len(test_images)} images of size {test_images[0].shape}")

    logging.info("Composite datasets created successfully!")


if __name__ == "__main__":
    main()
