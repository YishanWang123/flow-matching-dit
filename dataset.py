import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
from typing import Optional, Callable


class ImageNetSubset(Dataset):
    """ImageNet subset dataset."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        num_samples: Optional[int] = None,
    ):
        self.root = os.path.join(root, split)
        self.transform = transform

        # Collect all image paths
        self.images = []
        self.labels = []

        if os.path.exists(self.root):
            for class_idx, class_name in enumerate(sorted(os.listdir(self.root))):
                class_path = os.path.join(self.root, class_name)
                if os.path.isdir(class_path):
                    for img_name in os.listdir(class_path):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.images.append(os.path.join(class_path, img_name))
                            self.labels.append(class_idx)

        # Optionally limit the number of samples
        if num_samples is not None:
            self.images = self.images[:num_samples]
            self.labels = self.labels[:num_samples]

        print(f"Loaded {len(self.images)} images from {self.root}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(image_size: int = 32, is_train: bool = True):
    """Get data transforms for training or validation."""
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])


def create_dataloaders(
    data_path: str,
    image_size: int = 32,
    batch_size: int = 32,
    num_workers: int = 4,
    num_train_samples: Optional[int] = None,
    num_val_samples: Optional[int] = None,
):
    """Create training and validation dataloaders."""

    train_dataset = ImageNetSubset(
        root=data_path,
        split="train",
        transform=get_transforms(image_size, is_train=True),
        num_samples=num_train_samples,
    )

    val_dataset = ImageNetSubset(
        root=data_path,
        split="val",
        transform=get_transforms(image_size, is_train=False),
        num_samples=num_val_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def create_cifar10_dataloaders(
    image_size: int = 32,
    batch_size: int = 32,
    num_workers: int = 4,
    num_train_samples: Optional[int] = None,
    num_val_samples: Optional[int] = None,
):
    """Create CIFAR-10 training and validation dataloaders (real small dataset for validation)."""
    import torchvision.datasets as datasets

    # Download and load CIFAR-10
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=get_transforms(image_size, is_train=True),
    )

    val_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=get_transforms(image_size, is_train=False),
    )

    # Optionally limit number of samples for faster validation
    if num_train_samples is not None:
        train_dataset.data = train_dataset.data[:num_train_samples]
        train_dataset.targets = train_dataset.targets[:num_train_samples]
    if num_val_samples is not None:
        val_dataset.data = val_dataset.data[:num_val_samples]
        val_dataset.targets = val_dataset.targets[:num_val_samples]

    print(f"Loaded CIFAR-10: {len(train_dataset)} train samples, {len(val_dataset)} val samples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def create_synthetic_dataset(
    num_samples: int = 10000,
    image_size: int = 32,
    batch_size: int = 32,
):
    """Create a synthetic dataset for testing when ImageNet is not available."""

    class SyntheticDataset(Dataset):
        def __init__(self, num_samples, image_size):
            self.num_samples = num_samples
            self.image_size = image_size

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # Generate random images
            image = torch.randn(3, self.image_size, self.image_size)
            # Random label (0-999 for ImageNet classes)
            label = torch.randint(0, 1000, (1,)).item()
            return image, label

    dataset = SyntheticDataset(num_samples, image_size)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    return train_loader, val_loader
