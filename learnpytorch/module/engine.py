"""
Contains functions for training and testing a PyTorch model.
"""
import torch

from torch import nn
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device
):
    """
    Trains a PyTorch model for a single epoch.
    
    Sets the model to training mode.
    Performs all steps of training - forward pass, loss calculation, and optimizer step.

    Args:
        model: a PyTorch model to be trained.
        dataloader: a DataLoader for training data.
        loss_fn: a loss function to minimize.
        optimizer: an optimizer to help minimize the loss function.
        device: the device on which the model must be trained.

    Returns:
        A tuple of the form (train_loss, train_accuracy).
    """
    model.train()
    train_loss, train_acc = 0, 0
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc

def test_step(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        device: torch.device
):
    """
    Tests a PyTorch model on provided testing data.

    Args:
        model: a PyTorch model to be tested.
        dataloader: a DataLoader for testing data.
        loss_fn: a loss function to be used to evaluate the model.
        device: the device on which the model must be tested.

    Returns:
        A tuple of the form (test_loss, test_accuracy).
    """
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            test_pred = model(X)
            test_loss += loss_fn(test_pred, y)
            test_pred_class = torch.argmax(torch.softmax(test_pred, dim=1), dim=1)
            test_acc += (test_pred_class == y).sum().item() / len(test_pred)
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(
        model: nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        device: torch.device
) -> Dict[str, List]:
    """
    Trains and tests a PyTorch model.

    Passes a PyTorch model through train_step() and test_step() for the provided number of epochs.
    Calculates and returns evaluation metrics.

    Args:
        model: a PyTorch model.
        train_dataloader: a DataLoader for training data.
        test_dataloader: a DataLoader for testing data.
        loss_fn: a loss function to calculate loss.
        optimizer: an optimizer to help minimize the loss during training.
        epochs: an integer indiciating the number of times the model is to be trained.
        device: a device on which the model has to be trained and tested.

    Returns:
        Evaluation metrics as a dictionary of the below form.
        {
            train_loss: [...],
            train_acc: [...],
            test_loss: [...],
            test_acc: [...]
        }
    """
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model,
            train_dataloader,
            loss_fn,
            optimizer,
            device
        )
        test_loss, test_acc = test_step(
            model,
            test_dataloader,
            loss_fn,
            device
        )
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        print(f"Epoch: {(epoch + 1)} | Train loss: {train_loss:.3f} | Train accuracy: {train_acc:.3f} | Test loss: {test_loss:.3f} | Test accuracy: {test_acc:.3f}")
    return results