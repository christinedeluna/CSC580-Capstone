"""
CSC580 - CTA6
Prepare Cats vs. Dogs Image Data

Author: Christine DeLuna

Objective:
Convert the TensorFlow Datasets (TFDS) Cats vs. Dogs dataset into
local cat and dog image directories compatible with the file-based
NumPy/PIL workflow provided in the CSC580 assignment.

Dataset:
TensorFlow Datasets cats_vs_dogs
Total images: 23,262

The assignment originally specifies the Kaggle Dogs vs. Cats dataset.
Direct Kaggle access returned HTTP 403 despite authentication and
acceptance of the competition rules. TFDS provides a filtered version
of the same dataset with corrupted source images removed.
"""

# =====================================================
# Import Required Libraries
# =====================================================

from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds


# =====================================================
# Configuration
# =====================================================

# Store the prepared images inside this assignment directory.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

CAT_DIR = DATA_DIR / "cats"
DOG_DIR = DATA_DIR / "dogs"


# =====================================================
# Create Output Directories
# =====================================================

CAT_DIR.mkdir(parents=True, exist_ok=True)
DOG_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("CSC580 - Prepare Cats vs. Dogs Dataset")
print("=" * 60)

print("\nOutput directories:")
print("Cats:", CAT_DIR)
print("Dogs:", DOG_DIR)


# =====================================================
# Load Cached TensorFlow Dataset
# =====================================================

print("\nLoading cached Cats vs. Dogs dataset...")

dataset, info = tfds.load(
    "cats_vs_dogs",
    split="train",
    as_supervised=True,
    with_info=True,
    shuffle_files=False
)

class_names = info.features["label"].names
total_examples = info.splits["train"].num_examples

print("\nDataset loaded successfully.")
print("Total examples:", total_examples)
print("Class labels:", class_names)


# =====================================================
# Export Images
# =====================================================

print("\nExporting images...")

cat_count = 0
dog_count = 0

for index, (image, label) in enumerate(dataset):

    label_value = int(label.numpy())

    # TFDS labels:
    # 0 = cat
    # 1 = dog

    if label_value == 0:

        output_path = CAT_DIR / f"cat.{cat_count}.jpg"
        cat_count += 1

    else:

        output_path = DOG_DIR / f"dog.{dog_count}.jpg"
        dog_count += 1

    # Encode the TensorFlow image tensor as JPEG.
    encoded_image = tf.io.encode_jpeg(image)

    tf.io.write_file(
        str(output_path),
        encoded_image
    )

    # Display progress every 1,000 images.
    if (index + 1) % 1000 == 0:

        print(
            f"Processed {index + 1:,} "
            f"of {total_examples:,} images..."
        )


# =====================================================
# Verify Export
# =====================================================

exported_cats = len(list(CAT_DIR.glob("*.jpg")))
exported_dogs = len(list(DOG_DIR.glob("*.jpg")))

exported_total = exported_cats + exported_dogs

print("\n" + "=" * 60)
print("DATASET PREPARATION COMPLETE")
print("=" * 60)

print(f"\nCats exported:  {exported_cats:,}")
print(f"Dogs exported:  {exported_dogs:,}")
print(f"Total exported: {exported_total:,}")

print("\nExpected total:", f"{total_examples:,}")

if exported_total == total_examples:

    print("\nVerification: PASS")

else:

    print("\nVerification: WARNING")
    print(
        "Exported image count does not match "
        "the TFDS dataset count."
    )

print("\nPrepared dataset location:")
print(DATA_DIR)

print("=" * 60)