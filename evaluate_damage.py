"""
Phase 4: Measure per-image damage caused by INT8 quantization.

Compares the original FP32 model from Phase 1 with the complete
quantized INT8 model produced by compress.py.

For each of the same 3,000 training images, records:
- true label
- original prediction
- original true-class confidence
- quantized prediction
- quantized true-class confidence
- whether prediction flipped
- confidence drop
"""

import csv
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


# ============================================================
# Configuration
# ============================================================

DATA_DIR = "data"
RESULTS_DIR = "results"

MODEL_PATH = os.path.join(
    RESULTS_DIR,
    "model.pt",
)

QUANTIZED_MODEL_PATH = os.path.join(
    RESULTS_DIR,
    "model_int8.pt",
)

OUTPUT_PATH = os.path.join(
    RESULTS_DIR,
    "damage.csv",
)

BATCH_SIZE = 64


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


# ============================================================
# Dataset with stable image IDs
# ============================================================

class IndexedDataset(Dataset):
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
# Load original FP32 model
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False,
)

subset_indices = checkpoint["subset_indices"]

original_model = SmallCNN()

original_model.load_state_dict(
    checkpoint["model_state_dict"]
)

original_model.eval()

print("Original FP32 model loaded.")


# ============================================================
# Load complete INT8 model
# ============================================================

quantized_model = torch.load(
    QUANTIZED_MODEL_PATH,
    map_location="cpu",
    weights_only=False,
)

quantized_model.eval()

print("Complete INT8 model loaded.")


# ============================================================
# Dataset
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

evaluation_dataset = IndexedDataset(
    base_dataset,
    subset_indices,
)

evaluation_loader = DataLoader(
    evaluation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print(
    "Images to evaluate:",
    len(evaluation_dataset),
)


# ============================================================
# Evaluate both models
# ============================================================

results = []

with torch.no_grad():

    for images, labels, image_ids in evaluation_loader:

        # Original model
        original_logits = original_model(images)

        # Quantized model
        quantized_logits = quantized_model(images)

        original_probs = torch.softmax(
            original_logits,
            dim=1,
        )

        quantized_probs = torch.softmax(
            quantized_logits,
            dim=1,
        )

        original_predictions = (
            original_probs.argmax(dim=1)
        )

        quantized_predictions = (
            quantized_probs.argmax(dim=1)
        )

        for i in range(len(images)):

            true_label = int(labels[i])

            original_prediction = int(
                original_predictions[i]
            )

            quantized_prediction = int(
                quantized_predictions[i]
            )

            original_confidence = float(
                original_probs[
                    i,
                    true_label,
                ].item()
            )

            quantized_confidence = float(
                quantized_probs[
                    i,
                    true_label,
                ].item()
            )

            confidence_drop = (
                original_confidence
                - quantized_confidence
            )

            flipped = int(
                original_prediction
                != quantized_prediction
            )

            results.append([
                int(image_ids[i]),
                true_label,
                original_prediction,
                original_confidence,
                quantized_prediction,
                quantized_confidence,
                flipped,
                confidence_drop,
            ])


# ============================================================
# Save damage results
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True,
)

with open(
    OUTPUT_PATH,
    "w",
    newline="",
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "image_id",
        "true_label",
        "original_prediction",
        "original_confidence",
        "quantized_prediction",
        "quantized_confidence",
        "flipped",
        "confidence_drop",
    ])

    writer.writerows(results)


# ============================================================
# Summary
# ============================================================

num_images = len(results)

num_flipped = sum(
    row[6]
    for row in results
)

flip_rate = (
    num_flipped / num_images
)

print()
print("Damage evaluation complete.")
print("Images evaluated:", num_images)
print("Prediction flips:", num_flipped)
print(
    f"Flip rate: {flip_rate * 100:.2f}%"
)
print("Output:", OUTPUT_PATH)