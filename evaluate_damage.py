"""
Phase 4: Measure per-image damage caused by INT8 quantization.

Compares the original FP32 model with the serialized TorchScript
INT8 model on the exact same 3,000 CIFAR-10 images.

For every image, records:
- image_id
- true_label
- original_prediction
- original_confidence
- quantized_prediction
- quantized_confidence
- flipped
- confidence_drop

Output:
    results/damage.csv
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
    "model_int8_scripted.pt",
)

OUTPUT_PATH = os.path.join(
    RESULTS_DIR,
    "damage.csv",
)

BATCH_SIZE = 64


# ============================================================
# Original FP32 model
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
# Dataset wrapper with stable image IDs
# ============================================================

class IndexedDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        image, label = self.dataset[
            self.indices[position]
        ]

        image_id = int(
            self.indices[position]
        )

        return image, label, image_id


# ============================================================
# Load original FP32 model
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False,
)

subset_indices = checkpoint[
    "subset_indices"
]

original_model = SmallCNN()

original_model.load_state_dict(
    checkpoint["model_state_dict"]
)

original_model.eval()

print("Original FP32 model loaded.")


# ============================================================
# Load serialized TorchScript INT8 model
# ============================================================

quantized_model = torch.jit.load(
    QUANTIZED_MODEL_PATH,
    map_location="cpu",
)

quantized_model.eval()

print(
    "TorchScript INT8 model loaded."
)


# ============================================================
# Load CIFAR-10
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
# Three-image sanity check
# ============================================================

sanity_images, sanity_labels, sanity_ids = next(
    iter(evaluation_loader)
)

sanity_images = sanity_images[:3]
sanity_labels = sanity_labels[:3]
sanity_ids = sanity_ids[:3]

with torch.no_grad():

    fp32_sanity_logits = original_model(
        sanity_images
    )

    int8_sanity_logits = quantized_model(
        sanity_images
    )

fp32_sanity_probs = torch.softmax(
    fp32_sanity_logits,
    dim=1,
)

int8_sanity_probs = torch.softmax(
    int8_sanity_logits,
    dim=1,
)

fp32_sanity_predictions = (
    fp32_sanity_probs.argmax(dim=1)
)

int8_sanity_predictions = (
    int8_sanity_probs.argmax(dim=1)
)

print()
print("3-image sanity check:")

for i in range(3):

    image_id = int(
        sanity_ids[i]
    )

    true_label = int(
        sanity_labels[i]
    )

    fp32_prediction = int(
        fp32_sanity_predictions[i]
    )

    int8_prediction = int(
        int8_sanity_predictions[i]
    )

    fp32_confidence = float(
        fp32_sanity_probs[
            i,
            true_label,
        ].item()
    )

    int8_confidence = float(
        int8_sanity_probs[
            i,
            true_label,
        ].item()
    )

    print(
        f"Image {image_id}: "
        f"True={true_label}, "
        f"FP32={fp32_prediction} "
        f"(confidence={fp32_confidence:.4f}), "
        f"INT8={int8_prediction} "
        f"(confidence={int8_confidence:.4f})"
    )


# ============================================================
# Full damage evaluation
# ============================================================

results = []

with torch.no_grad():

    for images, labels, image_ids in evaluation_loader:

        # ----------------------------------------------------
        # FP32 inference
        # ----------------------------------------------------

        fp32_logits = original_model(
            images
        )

        # ----------------------------------------------------
        # INT8 inference
        # ----------------------------------------------------

        int8_logits = quantized_model(
            images
        )

        # ----------------------------------------------------
        # Convert logits to probabilities
        # ----------------------------------------------------

        fp32_probs = torch.softmax(
            fp32_logits,
            dim=1,
        )

        int8_probs = torch.softmax(
            int8_logits,
            dim=1,
        )

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        fp32_predictions = (
            fp32_probs.argmax(dim=1)
        )

        int8_predictions = (
            int8_probs.argmax(dim=1)
        )

        # ----------------------------------------------------
        # Per-image records
        # ----------------------------------------------------

        for i in range(len(images)):

            image_id = int(
                image_ids[i]
            )

            true_label = int(
                labels[i]
            )

            original_prediction = int(
                fp32_predictions[i]
            )

            quantized_prediction = int(
                int8_predictions[i]
            )

            original_confidence = float(
                fp32_probs[
                    i,
                    true_label,
                ].item()
            )

            quantized_confidence = float(
                int8_probs[
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
                image_id,
                true_label,
                original_prediction,
                original_confidence,
                quantized_prediction,
                quantized_confidence,
                flipped,
                confidence_drop,
            ])


# ============================================================
# Save CSV
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
# Summary statistics
# ============================================================

num_images = len(results)

num_flipped = sum(
    row[6]
    for row in results
)

flip_rate = (
    num_flipped / num_images
)

confidence_drops = [
    row[7]
    for row in results
]

confidence_tensor = torch.tensor(
    confidence_drops,
    dtype=torch.float64,
)

mean_confidence_drop = float(
    confidence_tensor.mean().item()
)

median_confidence_drop = float(
    confidence_tensor.median().item()
)


# ============================================================
# Final output
# ============================================================

print()
print(
    "Damage evaluation complete."
)

print(
    "Total images evaluated:",
    num_images,
)

print(
    "Number flipped:",
    num_flipped,
)

print(
    f"Flip rate: "
    f"{flip_rate * 100:.2f}%"
)

print(
    f"Mean confidence drop: "
    f"{mean_confidence_drop:.6f}"
)

print(
    f"Median confidence drop: "
    f"{median_confidence_drop:.6f}"
)

print(
    "Output:",
    OUTPUT_PATH,
)