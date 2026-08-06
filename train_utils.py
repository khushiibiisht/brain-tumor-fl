# train_utils.py
# Contains the train and test functions used by each FL client
# These run locally on each hospital's own MRI data

import torch
import torch.nn as nn
from tqdm import tqdm


# ─────────────────────────────────────────
# TRAINING FUNCTION
# Runs one full epoch on local data
# Called by each hospital client every FL round
# ─────────────────────────────────────────
def train(model, dataloader, device, epochs=1):
    """
    Trains the model for a given number of epochs.

    model      — the ResNet-18
    dataloader — local hospital's MRI images
    device     — 'cuda' or 'cpu'
    epochs     — how many passes over local data per FL round
                 (we use 1 — enough for federated learning)
    """

    # Loss function — CrossEntropy is standard for classification
    # It measures how wrong the model's predictions are
    # Higher loss = more wrong, Lower loss = more right
    criterion = nn.CrossEntropyLoss()

    # Optimizer — Adam adjusts model weights to reduce loss
    # lr=0.001 is the learning rate — how big each adjustment step is
    # we only optimize params that aren't frozen (layer4 + fc)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001
    )

    # Set model to training mode
    # This activates Dropout and BatchNorm training behaviour
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        # tqdm wraps the dataloader to show a progress bar
        loop = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')

        for images, labels in loop:
            # Move data to GPU
            images = images.to(device)
            labels = labels.to(device)

            # Zero out gradients from previous step
            # (PyTorch accumulates gradients by default)
            optimizer.zero_grad()

            # Forward pass — send images through ResNet-18
            outputs = model(images)

            # Calculate how wrong the predictions are
            loss = criterion(outputs, labels)

            # Backward pass — calculate gradients
            # (this is where skip connections help gradients flow)
            loss.backward()

            # Update weights based on gradients
            optimizer.step()

            # Track statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar with live stats
            loop.set_postfix(
                loss=f'{running_loss/len(dataloader):.3f}',
                acc=f'{100.*correct/total:.1f}%'
            )

    # Return final loss and accuracy for this epoch
    final_loss = running_loss / len(dataloader)
    final_acc = correct / total
    return final_loss, final_acc


# ─────────────────────────────────────────
# TESTING FUNCTION
# Evaluates model on test data
# Called after each FL round to measure global accuracy
# ─────────────────────────────────────────
def test(model, dataloader, device):
    """
    Evaluates the model — no training happens here.
    Returns loss and accuracy.
    """

    criterion = nn.CrossEntropyLoss()

    # Set model to evaluation mode
    # This disables Dropout and uses running stats for BatchNorm
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    # torch.no_grad() tells PyTorch not to calculate gradients
    # We don't need them for evaluation — saves memory and speed
    with torch.no_grad():
        loop = tqdm(dataloader, desc='Testing')

        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            loop.set_postfix(
                loss=f'{running_loss/len(dataloader):.3f}',
                acc=f'{100.*correct/total:.1f}%'
            )

    final_loss = running_loss / len(dataloader)
    final_acc = correct / total
    return final_loss, final_acc


# ─────────────────────────────────────────
# DATASET LOADER
# Loads MRI images from a folder
# Applies transforms to prepare for ResNet-18
# ─────────────────────────────────────────
def get_dataloader(data_dir, batch_size=16, shuffle=True):
    """
    Loads images from a folder and returns a DataLoader.

    data_dir   — path to folder containing class subfolders
    batch_size — how many images per batch (16 is safe for 4GB VRAM)
    shuffle    — randomise order (True for training, False for testing)
    """
    from torchvision import transforms, datasets

    # Transforms prepare raw images for ResNet-18
    transform = transforms.Compose([
        # Resize all MRI scans to 224x224
        # ResNet-18 expects this exact size
        transforms.Resize((224, 224)),

        # Data augmentation — only for training
        # Creates slightly varied versions of images
        # so model learns to generalise, not memorise
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),

        # Convert PIL image to PyTorch tensor
        transforms.ToTensor(),

        # Normalise using ImageNet mean and std
        # ResNet-18 pretrained weights expect this exact normalisation
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # ImageFolder automatically reads class subfolders
    # glioma/ → class 0, meningioma/ → class 1, etc.
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # DataLoader batches the images and handles shuffling
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0       # 0 = no multiprocessing (safe on Windows)
    )

    print(f'Loaded {len(dataset)} images from {data_dir}')
    print(f'Classes: {dataset.classes}')
    return dataloader