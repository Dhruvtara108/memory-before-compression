"""
Phase 3: Post-training static INT8 quantization.

Loads the trained CNN checkpoint, calibrates the model using
a subset of the training images, converts the model to INT8,
verifies INT8 inference, and saves the complete quantized model.
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
    "model_int8.pt",
)

CALIBRATION_IMAGES = 300
BATCH_SIZE = 64
SEED = 42


# ============================================================
# Reproducibility
# ============================================================

torch.manual_seed(SEED)


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
# Load original FP32 model
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False,
)

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

print(
    "CIFAR-10 training images:",
    len(dataset),
)


# ============================================================
# Reproduce the exact Phase 1 subset
# ============================================================

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
# Configure quantization
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

quantized_model.eval()

print("INT8 quantization complete.")


# ============================================================
# Verify INT8 inference on real CIFAR-10 images
# ============================================================

test_images, test_labels = next(
    iter(calibration_loader)
)

test_images = test_images[:5]
test_labels = test_labels[:5]

with torch.no_grad():
    int8_outputs = quantized_model(test_images)

print(
    "INT8 inference successful."
)

print(
    "Test images:",
    test_images.shape[0],
)

print(
    "Output shape:",
    int8_outputs.shape,
)

print(
    "Predictions:",
    int8_outputs.argmax(dim=1).tolist(),
)

print(
    "True labels:",
    test_labels.tolist(),
)


# ============================================================
# Save COMPLETE quantized model
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True,
)

torch.save(
    quantized_model,
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