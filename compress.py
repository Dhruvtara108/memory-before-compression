"""
Phase 3: Post-training static INT8 quantization.

Loads the trained CNN checkpoint, calibrates the model using
training images, converts the model to an INT8 quantized model,
and saves the compressed checkpoint.
"""

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
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

SEED = 42
CALIBRATION_IMAGES = 300


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
# Load model
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
)

model = SmallCNN()

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Original model loaded.")


# ============================================================
# Dataset
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform,
)

subset_indices = checkpoint["subset_indices"]

calibration_indices = subset_indices[
    :CALIBRATION_IMAGES
]

calibration_dataset = Subset(
    dataset,
    calibration_indices,
)

calibration_loader = DataLoader(
    calibration_dataset,
    batch_size=64,
    shuffle=False,
)

print(
    "Calibration images:",
    len(calibration_dataset),
)


# ============================================================
# Quantization configuration
# ============================================================

torch.backends.quantized.engine = "x86"

model.qconfig = torch.ao.quantization.get_default_qconfig(
    "x86"
)

# Prepare model for calibration.
torch.ao.quantization.prepare(
    model,
    inplace=True,
)


# ============================================================
# Calibration
# ============================================================

print("Starting calibration...")

with torch.no_grad():

    for images, _ in calibration_loader:
        model(images)

print("Calibration complete.")


# ============================================================
# Convert to INT8
# ============================================================

quantized_model = torch.ao.quantization.convert(
    model,
    inplace=False,
)

print("INT8 quantization complete.")


# ============================================================
# Save quantized model
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True,
)

torch.save(
    quantized_model.state_dict(),
    QUANTIZED_MODEL_PATH,
)

print(
    "Quantized model:",
    QUANTIZED_MODEL_PATH,
)


# ============================================================
# Compare model sizes
# ============================================================

original_size = os.path.getsize(
    MODEL_PATH
)

quantized_size = os.path.getsize(
    QUANTIZED_MODEL_PATH
)

print()
print(
    f"Original checkpoint size: "
    f"{original_size / 1024:.2f} KB"
)

print(
    f"Quantized checkpoint size: "
    f"{quantized_size / 1024:.2f} KB"
)

print(
    f"Size reduction: "
    f"{(1 - quantized_size / original_size) * 100:.2f}%"
)