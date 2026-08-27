"""
CSC580 - CTA6
Cats vs. Dogs Dataset Setup

Loads the Cats vs. Dogs image dataset using TensorFlow Datasets.

The original assignment references the Kaggle Dogs vs. Cats
competition. Direct Kaggle access returned HTTP 403 despite
authentication and acceptance of the competition rules.

An older TensorFlow-hosted ZIP distribution also returned HTTP 403.

TensorFlow Datasets is therefore used as the reproducible
dataset-loading method for this project.
"""

# =====================================================
# Import Libraries
# =====================================================

import tensorflow as tf
import tensorflow_datasets as tfds

# =====================================================
# Display Environment
# =====================================================

print("=" * 60)
print("CSC580 - Cats vs. Dogs Dataset Setup")
print("=" * 60)

print("\nTensorFlow Version:", tf.__version__)
print("TensorFlow Datasets Version:", tfds.__version__)

# =====================================================
# Load Cats vs. Dogs Dataset
# =====================================================

print("\nDownloading and preparing Cats vs. Dogs dataset...")

dataset, info = tfds.load(
    "cats_vs_dogs",
    split="train",
    as_supervised=True,
    with_info=True
)

# =====================================================
# Display Dataset Information
# =====================================================

print("\nDataset loaded successfully.")

print("\nDataset Information")
print("-" * 60)

print("Dataset name:", info.name)
print("Total examples:", info.splits["train"].num_examples)

print("\nClass labels:")
print(info.features["label"].names)

print("\nImage specification:")
print(info.features["image"])

print("\nDataset setup complete.")
print("=" * 60)
print("\nDataset setup complete.")