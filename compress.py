"""
Phase 3: Post-training static INT8 quantization.

Loads the trained FP32 CNN, calibrates it using a subset of CIFAR-10,
converts it to INT8, serializes it using TorchScript, verifies inference,
and reports checkpoint size reduction.
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

SCRIPTED_MODEL_PATH = os.path.join(
    RESULTS_DIR,
    "model_int8_scripted.pt",
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
# Load original FP32 checkpoint
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False,
)

model = SmallCNN()

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=False,
)

model.eval()

print("Original model loaded.")


# ============================================================
# Load CIFAR-10
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
# Reproduce Phase 1 subset
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
# Prepare for static quantization
# ============================================================

torch.backends.quantized.engine = "x86"

model.qconfig = (
    torch.ao.quantization.get_default_qconfig("x86")
)

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

quantized_model.eval()

print("INT8 quantization complete.")


# ============================================================
# Real-image inference sanity check
# ============================================================

test_images, test_labels = next(
    iter(calibration_loader)
)

test_images = test_images[:5]

with torch.no_grad():

    test_outputs = quantized_model(
        test_images
    )

test_predictions = (
    test_outputs.argmax(dim=1)
)

print("INT8 inference successful.")
print(
    "Test images:",
    len(test_images),
)

print(
    "Output shape:",
    test_outputs.shape,
)

print(
    "Predictions:",
    test_predictions.tolist(),
)

print(
    "True labels:",
    test_labels[:5].tolist(),
)


# ============================================================
# TorchScript serialization
# ============================================================

print("Creating TorchScript model...")

try:

    scripted_model = torch.jit.script(
        quantized_model
    )

    scripted_model.save(
        SCRIPTED_MODEL_PATH
    )

    serialization_method = "script"

    print(
        "TorchScript scripting successful."
    )

except Exception as script_error:

    print(
        "TorchScript scripting failed."
    )

    print(
        "Falling back to tracing..."
    )

    print(
        "Script error:",
        repr(script_error),
    )

    example_input = test_images[:1]

    scripted_model = torch.jit.trace(
        quantized_model,
        example_input,
    )

    scripted_model.save(
        SCRIPTED_MODEL_PATH
    )

    serialization_method = "trace"

    print(
        "TorchScript tracing successful."
    )


# ============================================================
# Reload TorchScript model
# ============================================================

loaded_scripted_model = torch.jit.load(
    SCRIPTED_MODEL_PATH,
    map_location="cpu",
)

loaded_scripted_model.eval()


# ============================================================
# Verify the SERIALIZED model
# ============================================================

with torch.no_grad():

    reloaded_outputs = (
        loaded_scripted_model(test_images)
    )

reloaded_predictions = (
    reloaded_outputs.argmax(dim=1)
)

print()
print(
    "TorchScript reload successful."
)

print(
    "Serialization method:",
    serialization_method,
)

print(
    "Reloaded output shape:",
    reloaded_outputs.shape,
)

print(
    "Reloaded predictions:",
    reloaded_predictions.tolist(),
)


# ============================================================
# Compare original INT8 model and reloaded model
# ============================================================

if torch.allclose(
    test_outputs,
    reloaded_outputs,
    atol=1e-4,
    rtol=1e-3,
):

    print(
        "Serialization verification: PASSED."
    )

else:

    raise RuntimeError(
        "Serialized TorchScript model "
        "does not reproduce the original "
        "INT8 model outputs."
    )


# ============================================================
# Compare file sizes
# ============================================================

original_size = os.path.getsize(
    MODEL_PATH
)

scripted_size = os.path.getsize(
    SCRIPTED_MODEL_PATH
)

reduction = (
    1 - scripted_size / original_size
) * 100


print()
print(
    "TorchScript INT8 model:",
    SCRIPTED_MODEL_PATH,
)

print(
    f"Original checkpoint size: "
    f"{original_size / 1024:.2f} KB"
)

print(
    f"TorchScript INT8 size: "
    f"{scripted_size / 1024:.2f} KB"
)

print(
    f"Size reduction: "
    f"{reduction:.2f}%"
)