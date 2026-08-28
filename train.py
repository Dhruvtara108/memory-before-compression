"""
Phase 1: Train a small CNN on a fixed CIFAR-10 subset.

The script:
1. Selects a reproducible 3,000-image subset of CIFAR-10.
2. Trains a small 4-layer CNN for 20 epochs.
3. Records one loss value for every image after every epoch.
4. Saves the trained model checkpoint.
5. Saves the selected image IDs for reproducibility.
"""

import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


# ============================================================
# Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Experiment configuration
# ============================================================

DATA_DIR = "data"
RESULTS_DIR = "results"

NUM_IMAGES = 3_000
NUM_EPOCHS = 20
BATCH_SIZE = 64

LEARNING_RATE = 0.001


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)
print("Seed:", SEED)
print("Images:", NUM_IMAGES)
print("Epochs:", NUM_EPOCHS)
print("Batch size:", BATCH_SIZE)
print("Learning rate:", LEARNING_RATE)


# ============================================================
# Dataset with stable image IDs
# ============================================================

class IndexedSubset(Dataset):
    """
    Wrap a subset so every sample returns its original image ID.
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        image, label = self.dataset[self.indices[position]]
        image_id = int(self.indices[position])

        return image, label, image_id


# ============================================================
# CIFAR-10 dataset
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
])

base_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform,
)

print("CIFAR-10 training images:", len(base_dataset))


# ============================================================
# Select reproducible subset
# ============================================================

rng = np.random.default_rng(SEED)

subset_indices = rng.choice(
    len(base_dataset),
    size=NUM_IMAGES,
    replace=False,
)

subset_indices = subset_indices.tolist()

print("Selected subset images:", len(subset_indices))
print("First 5 image IDs:", subset_indices[:5])


train_dataset = IndexedSubset(
    base_dataset,
    subset_indices,
)


# ============================================================
# DataLoaders
# ============================================================

# Shuffled loader is used for training.
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# Non-shuffled loader is used to evaluate every image
# consistently after each epoch.
logging_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print("Training batches:", len(train_loader))
print("Logging batches:", len(logging_loader))


# ============================================================
# Small CNN
# ============================================================

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


model = SmallCNN().to(DEVICE)

print("Model created successfully.")
print(model)


# ============================================================
# Loss function and optimizer
# ============================================================

# reduction="none" gives one loss value per image.
criterion = nn.CrossEntropyLoss(reduction="none")

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

print("Loss function created successfully.")
print("Optimizer created successfully.")


# ============================================================
# Prepare output directory
# ============================================================

os.makedirs(RESULTS_DIR, exist_ok=True)

loss_log_path = os.path.join(
    RESULTS_DIR,
    "per_sample_epoch_loss.csv",
)

checkpoint_path = os.path.join(
    RESULTS_DIR,
    "model.pt",
)

subset_path = os.path.join(
    RESULTS_DIR,
    "subset_indices.csv",
)


# ============================================================
# Save selected image IDs
# ============================================================

with open(subset_path, "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow(["image_id"])

    for image_id in subset_indices:
        writer.writerow([image_id])


# ============================================================
# Initialize loss log
# ============================================================

with open(loss_log_path, "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "image_id",
        "epoch",
        "loss",
    ])


# ============================================================
# Training
# ============================================================

for epoch in range(1, NUM_EPOCHS + 1):

    model.train()

    total_training_loss = 0.0
    total_samples = 0

    for images, labels, image_ids in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        individual_losses = criterion(
            outputs,
            labels,
        )

        # Backpropagation requires one scalar loss.
        loss = individual_losses.mean()

        loss.backward()
        optimizer.step()

        total_training_loss += (
            loss.item() * images.size(0)
        )

        total_samples += images.size(0)

    average_training_loss = (
        total_training_loss / total_samples
    )


    # ========================================================
    # Evaluate every training image after this epoch
    # ========================================================

    model.eval()

    epoch_losses = []

    with torch.no_grad():

        for images, labels, image_ids in logging_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            individual_losses = criterion(
                outputs,
                labels,
            )

            for image_id, sample_loss in zip(
                image_ids,
                individual_losses,
            ):
                epoch_losses.append(
                    (
                        int(image_id),
                        epoch,
                        float(sample_loss.item()),
                    )
                )


    # ========================================================
    # Append this epoch's per-image losses to CSV
    # ========================================================

    with open(loss_log_path, "a", newline="") as file:

        writer = csv.writer(file)

        writer.writerows(epoch_losses)


    print(
        f"Epoch {epoch:02d}/{NUM_EPOCHS} "
        f"- Training Loss: {average_training_loss:.4f} "
        f"- Logged Samples: {len(epoch_losses)}"
    )


# ============================================================
# Save final model
# ============================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "seed": SEED,
        "num_images": NUM_IMAGES,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "subset_indices": subset_indices,
    },
    checkpoint_path,
)


print()
print("Training complete.")
print("Loss log:", loss_log_path)
print("Model checkpoint:", checkpoint_path)
print("Subset IDs:", subset_path)