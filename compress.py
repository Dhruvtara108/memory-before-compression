"""
Phase 3: Post-training static INT8 quantization.

Loads the trained CNN checkpoint, calibrates the model using
a subset of the training images, converts the model to INT8,
and saves the quantized model.
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

CALIBRATION_IMAGES = 300
BATCH_SIZE = 64
SEED = 42


# ============================================================
# Quantizable CNN
# ============================================================

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.quant = torch.ao.quantization.QuantStub()

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

        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)

        x = self.features(x)

        x = torch.flatten(x, 1)

        x = self.classifier(x)

        x = self.dequant(x)

        return x


# ============================================================
# Load original model
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
)

# The original Phase 1 checkpoint was created from the same
# SmallCNN architecture without quantization stubs.
original_model = SmallCNN()

original_model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=False,
)

original_model.eval()

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
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print(
    "Calibration images:",
    len(calibration_dataset),
)


# ============================================================
# Prepare model for quantization
# ============================================================

torch.backends.quantized.engine = "x86"

original_model.qconfig = (
    torch.ao.quantization.get_default_qconfig("x86")
)

torch.ao.quantization.prepare(
    original_model,
    inplace=True,
)


# ============================================================
# Calibration
# ============================================================

print("Starting calibration...")

with torch.no_grad():

    for images, _ in calibration_loader:
        original_model(images)

print("Calibration complete.")


# ============================================================
# Convert to INT8
# ============================================================

quantized_model = torch.ao.quantization.convert(
    original_model,
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
# Compare checkpoint sizes
# ============================================================

original_size = os.path.getsize(
    MODEL_PATH
)

quantized_size = os.path.getsize(
    QUANTIZED_MODEL_PATH
)

reduction = (
    1 - quantized_size / original_size
) * 100

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
    f"{reduction:.2f}%"
)