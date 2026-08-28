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

Additional diagnostics:
1. Explicit confidence_drop definition.
2. Confidence-drop distribution.
3. Median CSL score for flipped vs non-flipped images.
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

CSL_PATH = os.path.join(
    RESULTS_DIR,
    "csl_scores.csv",
)

BATCH_SIZE = 64


# ============================================================
# Original FP32 model
# ============================================================

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(
            128,
            10,
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ============================================================
# Dataset wrapper with stable image IDs
# ============================================================

class IndexedDataset(Dataset):

    def __init__(
        self,
        dataset,
        indices,
    ):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(
        self,
        position,
    ):
        image, label = self.dataset[
            self.indices[position]
        ]

        image_id = int(
            self.indices[position]
        )

        return (
            image,
            label,
            image_id,
        )


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

print(
    "Original FP32 model loaded."
)


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

    fp32_sanity_logits = (
        original_model(
            sanity_images
        )
    )

    int8_sanity_logits = (
        quantized_model(
            sanity_images
        )
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
print(
    "3-image sanity check:"
)


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

    for (
        images,
        labels,
        image_ids,
    ) in evaluation_loader:

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

            # IMPORTANT:
            # Confidence is probability assigned to the
            # TRUE LABEL, not probability of the predicted
            # class.

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

            # Formula:
            #
            # confidence_drop =
            #     original true-label confidence
            #     -
            #     quantized true-label confidence

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
# Save damage CSV
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

    writer.writerows(
        results
    )


# ============================================================
# Basic summary statistics
# ============================================================

num_images = len(results)

num_flipped = sum(
    row[6]
    for row in results
)

flip_rate = (
    num_flipped
    / num_images
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
# Final Phase 4 summary
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


# ============================================================
# PHASE 4 DIAGNOSTICS
# ============================================================

print()
print("=" * 60)
print("PHASE 4 DIAGNOSTICS")
print("=" * 60)


# ============================================================
# 1. Confidence-drop definition
# ============================================================

print()
print(
    "1. Confidence-drop definition"
)

print(
    "Formula:"
)

print(
    "confidence_drop = "
    "original true-label confidence "
    "- quantized true-label confidence"
)

print(
    "Therefore:"
)

print(
    "Positive confidence_drop = "
    "INT8 reduced probability assigned to the true label."
)

print(
    "Negative confidence_drop = "
    "INT8 increased probability assigned to the true label."
)

print(
    "Confidence means probability assigned to the "
    "TRUE LABEL, not confidence in the model's predicted class."
)


# ============================================================
# 2. Confidence-drop distribution
# ============================================================

negative_count = sum(
    drop < 0
    for drop in confidence_drops
)

zero_to_005_count = sum(
    0 <= drop < 0.05
    for drop in confidence_drops
)

range_005_to_02_count = sum(
    0.05 <= drop < 0.20
    for drop in confidence_drops
)

above_or_equal_02_count = sum(
    drop >= 0.20
    for drop in confidence_drops
)


print()
print(
    "2. Confidence-drop distribution"
)

print(
    f"< 0:        "
    f"{negative_count:4d} "
    f"({negative_count / num_images * 100:.2f}%)"
)

print(
    f"0–0.05:     "
    f"{zero_to_005_count:4d} "
    f"({zero_to_005_count / num_images * 100:.2f}%)"
)

print(
    f"0.05–0.20:  "
    f"{range_005_to_02_count:4d} "
    f"({range_005_to_02_count / num_images * 100:.2f}%)"
)

print(
    f">= 0.20:    "
    f"{above_or_equal_02_count:4d} "
    f"({above_or_equal_02_count / num_images * 100:.2f}%)"
)

print(
    "Bucket total:",
    (
        negative_count
        + zero_to_005_count
        + range_005_to_02_count
        + above_or_equal_02_count
    ),
)


# ============================================================
# 3. CSL median: flipped vs non-flipped
# ============================================================

if not os.path.exists(CSL_PATH):

    raise FileNotFoundError(
        f"CSL file not found: {CSL_PATH}"
    )


csl_values = {}


with open(
    CSL_PATH,
    "r",
    newline="",
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        image_id = int(
            row["image_id"]
        )

        csl_score = float(
            row["csl_score"]
        )

        csl_values[image_id] = csl_score


flipped_csl = []
non_flipped_csl = []


for row in results:

    image_id = row[0]
    flipped = row[6]

    if image_id not in csl_values:

        raise RuntimeError(
            "Missing CSL score for "
            f"image_id={image_id}"
        )

    if flipped == 1:

        flipped_csl.append(
            csl_values[image_id]
        )

    else:

        non_flipped_csl.append(
            csl_values[image_id]
        )


if not flipped_csl:

    raise RuntimeError(
        "No flipped images found; "
        "cannot compute flipped-group median CSL."
    )


if not non_flipped_csl:

    raise RuntimeError(
        "No non-flipped images found; "
        "cannot compute non-flipped-group median CSL."
    )


flipped_csl_tensor = torch.tensor(
    flipped_csl,
    dtype=torch.float64,
)

non_flipped_csl_tensor = torch.tensor(
    non_flipped_csl,
    dtype=torch.float64,
)


flipped_median_csl = float(
    flipped_csl_tensor.median().item()
)

non_flipped_median_csl = float(
    non_flipped_csl_tensor.median().item()
)


print()
print(
    "3. Median CSL by prediction-flip group"
)

print(
    f"Flipped images:     "
    f"{len(flipped_csl):4d} images | "
    f"median CSL = "
    f"{flipped_median_csl:.6f}"
)

print(
    f"Non-flipped images: "
    f"{len(non_flipped_csl):4d} images | "
    f"median CSL = "
    f"{non_flipped_median_csl:.6f}"
)

print(
    "CSL group counts total:",
    len(flipped_csl)
    + len(non_flipped_csl),
)

print(
    "CSL images available:",
    len(csl_values),
)

print("=" * 60)