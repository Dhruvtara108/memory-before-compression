"""
Phase 4: Measure per-image damage caused by quantization.

For every image in the same 3,000-image subset used during training,
compare the original FP32 model with the quantized INT8 model.

Records:
- original prediction
- original confidence for the true class
- quantized prediction
- quantized confidence for the true class
- prediction flip
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
    "model_quantized.pt",
)

SUBSET_PATH = os.path.join(
    RESULTS_DIR,
    "subset_indices.csv",
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
    """
    Returns image, label, and the original CIFAR-10 image ID.
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
# Load checkpoint
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
)

subset_indices = checkpoint["subset_indices"]

print("Images to evaluate:", len(subset_indices))


# ============================================================
# Load original FP32 model
# ============================================================

original_model = SmallCNN()

original_model.load_state_dict(
    checkpoint["model_state_dict"]
)

original_model.eval()

print("Original FP32 model loaded.")


# ============================================================
# Reconstruct quantized model
# ============================================================

torch.backends.quantized.engine = "x86"

quantized_model = SmallCNN()

quantized_model.qconfig = torch.ao.quantization.get_default_qconfig(
    "x86"
)

torch.ao.quantization.prepare(
    quantized_model,
    inplace=True,
)


# Calibration is required before loading the quantized
# state dictionary because the quantized module structure
# must exist first.
transform = transforms.Compose([
    transforms.ToTensor(),
])

base_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform,
)

calibration_indices = subset_indices[:300]

calibration_dataset = torch.utils.data.Subset(
    base_dataset,
    calibration_indices,
)

calibration_loader = DataLoader(
    calibration_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


print("Calibrating quantized model...")

with torch.no_grad():
    for images, _ in calibration_loader:
        quantized_model(images)

quantized_state_dict = torch.load(
    QUANTIZED_MODEL_PATH,
    map_location="cpu",
)

quantized_model.load_state_dict(
    quantized_state_dict,
)

quantized_model = torch.ao.quantization.convert(
    quantized_model,
    inplace=False,
)
quantized_model.eval()

print("Quantized INT8 model loaded.")


# ============================================================
# Evaluation dataset
# ============================================================

evaluation_dataset = IndexedDataset(
    base_dataset,
    subset_indices,
)

evaluation_loader = DataLoader(
    evaluation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ============================================================
# Evaluate both models
# ============================================================

results = []

print("Evaluating 3,000 images...")

with torch.no_grad():

    for images, labels, image_ids in evaluation_loader:

        original_logits = original_model(images)
        quantized_logits = quantized_model(images)

        original_probabilities = torch.softmax(
            original_logits,
            dim=1,
        )

        quantized_probabilities = torch.softmax(
            quantized_logits,
            dim=1,
        )

        original_predictions = (
            original_probabilities.argmax(dim=1)
        )

        quantized_predictions = (
            quantized_probabilities.argmax(dim=1)
        )

        for index in range(len(images)):

            true_label = int(labels[index])

            original_prediction = int(
                original_predictions[index]
            )

            quantized_prediction = int(
                quantized_predictions[index]
            )

            original_confidence = float(
                original_probabilities[
                    index,
                    true_label,
                ].item()
            )

            quantized_confidence = float(
                quantized_probabilities[
                    index,
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
                int(image_ids[index]),
                true_label,
                original_prediction,
                original_confidence,
                quantized_prediction,
                quantized_confidence,
                flipped,
                confidence_drop,
            ])


# ============================================================
# Save results
# ============================================================

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

num_flipped = sum(
    row[6] for row in results
)

flip_rate = (
    num_flipped / len(results)
)

print()
print("Damage evaluation complete.")
print("Images evaluated:", len(results))
print("Prediction flips:", num_flipped)
print(
    f"Flip rate: {flip_rate * 100:.2f}%"
)
print("Output:", OUTPUT_PATH)