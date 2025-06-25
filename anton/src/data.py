import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from torch.nn.utils.rnn import pad_sequence
from torchvision import datasets, transforms
from torch.utils.data import random_split
from PIL import Image
import hashlib
import pickle
from tqdm import tqdm
import numpy as np
import random
import os


# TODO:
# 2. Wrap into <start> <end>
# 3. don't hardcode size 28.
# 4. Replace transforms.Normalize((0.1307,), (0.3081,)) with the real mean and std for the dataset.


PAD_TOKEN = 10
START_TOKEN = 11
END_TOKEN = 12


def patchify(img, patch_size=14):
    """
    Converts an image into flatenned patches.
    Args:
        img (Tensor):     shape (1, H, W);                        example (1, 28, 28)
        patch_size (int): width and height of each patch;         example 14
    Returns:
        Tensor: (num_patches, patch_dim) flattened patches image; example (4, 196)
    """
    # img: torch.Tensor with shape (1, H, W)
    patches = img.unfold(1, patch_size, patch_size) \
                 .unfold(2, patch_size, patch_size)
    # patches shape: (1, num_patches_h, num_patches_w, patch_size, patch_size)
    patches = patches.contiguous().view(-1, patch_size * patch_size)
    return patches  # shape: (num_patches, patch_dim)


def unpatchify(patches, patch_size=14):
    """
    Reconstruct an image from flattened patches.
    Args:
        patches (Tensor): shape (num_patches, patch_dim); example (4, 196)
        patch_size (int): width and height of each patch; example 14
    Returns:
        Tensor: (1, H, W) reconstructed image;            example (1, 28, 28)
    """
    num_patches, patch_dim = patches.shape
    assert patch_dim == patch_size ** 2, f"Patch dim mismatch: {patch_dim} != {patch_size}^2"

    # Infer grid size
    grid_size = int(np.sqrt(num_patches))
    assert grid_size ** 2 == num_patches, "Number of patches must be a perfect square"

    # (num_patches, patch_size, patch_size)
    patches = patches.view(grid_size, grid_size, patch_size, patch_size)

    # Reconstruct rows
    rows = [torch.cat([patches[i, j] for j in range(grid_size)], dim=1) for i in range(grid_size)]

    # Combine all rows
    full_img = torch.cat(rows, dim=0)  # (H, W)
    return full_img.unsqueeze(0)  # (1, H, W)


class PatchifiedMNIST(Dataset):
    def __init__(self, root, train=True, download=True, patch_size=14, transform=None):
        self.base_dataset = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transform if transform else transforms.ToTensor()
        )
        self.patch_size = patch_size

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]  # img: (1, 28, 28)
        patches = patchify(img, self.patch_size)  # patches: (4, 196)
        return patches, label


class CompositePatchifiedMNIST(Dataset):
    def __init__(self, root, train=True, download=True, transform=None,
                 canvas_size=(56, 56), patch_size=14,
                 num_digits=4, placement='grid', num_images=10000,
                 num_digits_range=None, cache=True, cache_dir=None):

        self.canvas_w, self.canvas_h = canvas_size
        self.patch_size = patch_size
        self.num_digits = num_digits
        self.num_digits_range = num_digits_range
        self.placement = placement
        self.num_images = num_images
        self.transform = transform if transform else transforms.ToTensor()
        self.cache = cache

        self.mnist = datasets.MNIST(root=root, train=train, download=download, transform=transforms.ToTensor())

        # Generate a hash to identify the cache
        self.cache_dir = cache_dir or os.path.join(root, 'MNIST')
        os.makedirs(self.cache_dir, exist_ok=True)

        self.cache_key = self._make_cache_key(canvas_size, patch_size, num_digits, num_digits_range,
                                              placement, num_images, train)
        self.cache_file = os.path.join(self.cache_dir, self.cache_key + '.pkl')

        if cache and os.path.exists(self.cache_file):
            print(f"Loading cached dataset: {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                self.samples = pickle.load(f)
        else:
            print(f"Generating composite dataset: {self.cache_file}")
            self.samples = self._precompute()
            if cache:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(self.samples, f)

    def _make_cache_key(self, *args):
        hash_input = str(args).encode()
        return hashlib.md5(hash_input).hexdigest()

    def _precompute(self):
        samples = []
        for _ in tqdm(range(self.num_images), desc="Generating composite MNIST"):
            img = Image.new('L', (self.canvas_w, self.canvas_h), color=0)
            labels = []

            num_digits = self.num_digits
            if self.num_digits_range:
                num_digits = random.randint(*self.num_digits_range)

            if self.placement == 'grid':
                grid_rows = int(np.floor(np.sqrt(num_digits)))
                grid_cols = int(np.ceil(num_digits / grid_rows))
                patch_w, patch_h = self.canvas_w // grid_cols, self.canvas_h // grid_rows

                for i in range(num_digits):
                    idx = random.randint(0, len(self.mnist) - 1)
                    digit, label = self.mnist[idx]
                    digit = transforms.Resize((patch_h, patch_w))(digit)
                    row, col = divmod(i, grid_cols)
                    x = col * patch_w
                    y = row * patch_h
                    img.paste(transforms.ToPILImage()(digit), (x, y))
                    labels.append(label)

            elif self.placement == 'random':
                grid_rows = int(np.floor(np.sqrt(num_digits)))
                grid_cols = int(np.ceil(num_digits / grid_rows))
                total_cells = grid_rows * grid_cols

                # Randomly choose cells where digits will go
                chosen_cells = random.sample(range(total_cells), num_digits)
                chosen_cells.sort()  # Keep left-to-right, top-to-bottom order

                for i, cell_index in enumerate(chosen_cells):
                    idx = random.randint(0, len(self.mnist) - 1)
                    digit_tensor, label = self.mnist[idx]
                    digit_img = transforms.ToPILImage()(digit_tensor)

                    row = cell_index // grid_cols
                    col = cell_index % grid_cols

                    # Compute base position for this cell
                    cell_w = self.canvas_w // grid_cols
                    cell_h = self.canvas_h // grid_rows
                    base_x = col * cell_w
                    base_y = row * cell_h

                    # Jitter (can push image partially outside cell)
                    max_x_jitter = max(cell_w - 28, 0)
                    max_y_jitter = max(cell_h - 28, 0)
                    jitter_x = random.randint(0, max_x_jitter) if max_x_jitter > 0 else 0
                    jitter_y = random.randint(0, max_y_jitter) if max_y_jitter > 0 else 0

                    x = base_x + jitter_x
                    y = base_y + jitter_y

                    img.paste(digit_img, (x, y))
                    labels.append(label)
            # TODO Else show error

            tensor_img = self.transform(img)  # (1, H, W)
            patches = patchify(tensor_img, self.patch_size)  # (num_patches, patch_dim)
            samples.append((patches, torch.tensor(labels)))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def padded_collate_fn(batch):
    inputs, targets = zip(*batch)  # list of (input, target)

    # Collate input tensors (images or patchified inputs)
    inputs = default_collate(inputs)

    # Convert all targets to tensors
    targets = [torch.tensor(t) if not isinstance(t, torch.Tensor) else t for t in targets]

    # If all targets are scalar (single value), stack them
    if all(t.ndim == 0 for t in targets):
        targets = torch.stack(targets)  # Shape: (B,)
    else:
        # Otherwise treat them as sequences. Include <START> and <END> tokens and pad
        processed_targets = []
        for seq in targets:
            seq_tensor = torch.tensor([START_TOKEN] + list(seq) + [END_TOKEN], dtype=torch.long)
            processed_targets.append(seq_tensor)

        targets = pad_sequence(processed_targets, batch_first=True, padding_value=PAD_TOKEN)  # Shape: (B, T)

    return inputs, targets


def load_mnist_dataloaders(cache_dir, batch_size=64, valid_fraction=0.2, patch_size=14,
                            seed=42, num_workers=2, composite_mode=False,
                            canvas_size=(56, 56), num_digits=4, placement='grid',
                            num_digits_range=None, num_images=10000, num_images_test=2000):

    # TODO Do not hardcode these parameters
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if composite_mode:
        full_dataset = CompositePatchifiedMNIST(
            root=cache_dir,
            train=True,
            download=True,
            transform=transform,
            canvas_size=canvas_size,
            num_digits=num_digits,
            placement=placement,
            num_digits_range=num_digits_range,
            num_images=num_images
        )
        test_dataset = CompositePatchifiedMNIST(
            root=cache_dir,
            train=False,
            download=True,
            transform=transform,
            canvas_size=canvas_size,
            num_digits=num_digits,
            placement=placement,
            num_digits_range=num_digits_range,
            num_images=num_images_test
        )
    else:
        full_dataset = PatchifiedMNIST(root=cache_dir, train=True, download=True, patch_size=patch_size, transform=transform)
        test_dataset = PatchifiedMNIST(root=cache_dir, train=False, download=True, patch_size=patch_size, transform=transform)

    valid_size = int(valid_fraction * len(full_dataset))
    train_size = len(full_dataset) - valid_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, valid_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=padded_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=padded_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=padded_collate_fn)

    return train_loader, val_loader, test_loader
