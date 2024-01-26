"""
Contains functionality for creating PyTorch DataLoaders for image classification data.
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = 0

def create_dataloaders(
        train_dir: str,
        test_dir: str,
        transform: transforms.Compose,
        batch_size: int,
        num_workers: int = NUM_WORKERS
):
    """
    Creates training and testing Dataloaders.

    Takes in a training directory and a testing directory and creates a PyTorch Datalaoder for each.

    Args:
        train_dir: path to training directory.
        test_dir: path to testing directory.
        transform: torchvision transforms that have to be performed on data.
        batch_size: number of images in each batch.
        num_workers: number of workers per Dataloader.

    Returns:
        A tuple (train_dataloader, test_dataloader, class_names).
        Here class_names is the list of valid classes the data is paritioned into.
    """
    # Use ImageFolder to create Datasets
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    # Get class names
    class_names = train_data.classes

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataloader, test_dataloader, class_names