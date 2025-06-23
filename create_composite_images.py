import torch
import numpy as np
from torchvision import datasets
from torchvision.transforms import v2
import random
import os
from collections import defaultdict
import utils
from tqdm import tqdm
import logging


def create_composite_image(mnist_images, mnist_labels, num_images):
    """Create a 56x56 composite image with num_images MNIST digits"""
    composite = torch.zeros(1, 56, 56)
    labels = []

    # Define the 4 quadrant positions (top-left corners)
    positions = [(0, 0), (0, 28), (28, 0), (28, 28)]

    # Randomly select which positions to fill
    selected_positions = random.sample(positions, num_images)

    # Randomly select MNIST images
    indices = random.choices(range(len(mnist_images)), k=num_images)

    for i, pos in enumerate(selected_positions):
        idx = indices[i]
        # Place the 28x28 MNIST image at the selected position
        y, x = pos
        composite[0, y : y + 28, x : x + 28] = mnist_images[idx][
            0
        ]  # MNIST images are (1, 28, 28)
        labels.append(mnist_labels[idx])

    # Pad labels with -1 for empty positions (so we always have 4 labels)
    while len(labels) < 4:
        labels.append(-1)

    return composite, torch.tensor(labels)


def generate_composite_dataset(mnist_dataset, size, distribution):
    """Generate composite dataset with given size and distribution"""
    composite_images = []
    composite_labels = []

    # Convert MNIST data to lists for easier random access
    mnist_images = []
    mnist_labels = []
    for img, label in mnist_dataset:
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

        composite_img, composite_label = create_composite_image(
            mnist_images, mnist_labels, num_images
        )

        composite_images.append(composite_img)
        composite_labels.append(composite_label)

    return torch.stack(composite_images), torch.stack(composite_labels)


def split_dataset(images, labels, val_split=0.2):
    """Split dataset into training and validation sets"""
    dataset_size = len(images)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size

    # Create random indices
    indices = torch.randperm(dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_images = images[train_indices]
    train_labels = labels[train_indices]
    val_images = images[val_indices]
    val_labels = labels[val_indices]

    return train_images, train_labels, val_images, val_labels


def save_torch_dataset(images, labels, filepath):
    """Save dataset as torch tensors"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    torch.save({"images": images, "labels": labels}, filepath)

    logging.info(f"Saved dataset to {filepath}")


def log_dataset_stats(labels, dataset_name):
    """print statistics about dataset distribution"""
    counts = defaultdict(int)
    for label_set in labels:
        num_valid = (label_set != -1).sum().item()
        counts[num_valid] += 1

    logging.info(f"\n{dataset_name} distribution:")
    for num_imgs in sorted(counts.keys()):
        percentage = (counts[num_imgs] / len(labels)) * 100
        logging.info(
            f"  {num_imgs} images: {counts[num_imgs]} samples ({percentage:.1f}%)"
        )


def main():
    utils.setup_logging()
    logging.info("Downloading original MNIST dataset...")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Load original MNIST data
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

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
    all_train_images, all_train_labels = generate_composite_dataset(
        training_data, original_train_size, distribution
    )

    logging.info("Splitting training set into train/validation (80%/20%)...")
    train_images, train_labels, val_images, val_labels = split_dataset(
        all_train_images, all_train_labels, val_split=0.2
    )

    logging.info("Generating composite test dataset...")
    test_images, test_labels = generate_composite_dataset(
        test_data, test_size, distribution
    )

    # Save datasets
    logging.info("Saving datasets...")
    save_torch_dataset(train_images, train_labels, "data/composite_train.pt")
    save_torch_dataset(val_images, val_labels, "data/composite_val.pt")
    save_torch_dataset(test_images, test_labels, "data/composite_test.pt")

    # logging.info statistics
    logging.info("Dataset Statistics:")
    logging.info(
        f"Training set: {len(train_images)} images of size {train_images[0].shape}"
    )
    logging.info(
        f"Validation set: {len(val_images)} images of size {val_images[0].shape}"
    )
    logging.info(f"Test set: {len(test_images)} images of size {test_images[0].shape}")

    # logging.info distribution statistics
    log_dataset_stats(train_labels, "Training set")
    log_dataset_stats(val_labels, "Validation set")
    log_dataset_stats(test_labels, "Test set")

    logging.info("Composite datasets created successfully!")


if __name__ == "__main__":
    main()
