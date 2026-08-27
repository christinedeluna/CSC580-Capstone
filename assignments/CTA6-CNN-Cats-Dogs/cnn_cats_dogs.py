"""
CSC580 - CTA6
Convolutional Neural Network: Cats vs. Dogs

This project compares a fully connected neural network with a
convolutional neural network (CNN) for binary image classification.

The original assignment specifies the Kaggle Dogs vs. Cats dataset.
Due to current Kaggle access restrictions, the project uses the
TensorFlow Datasets distribution of the same dataset, containing
23,262 usable images after corrupted source images were removed.
"""

# -------------------------------------------------
# Imports
# -------------------------------------------------

from pathlib import Path
from collections import defaultdict

import glob
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# -------------------------------------------------
# Reproducibility
# -------------------------------------------------

SEED = 456

np.random.seed(SEED)
tf.random.set_seed(SEED)


# -------------------------------------------------
# Dataset Paths
# -------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

CAT_PATH = str(PROJECT_DIR / "data" / "cats" / "*")
DOG_PATH = str(PROJECT_DIR / "data" / "dogs" / "*")


# -------------------------------------------------
# Image Configuration
# -------------------------------------------------

# The assignment identifies 375 x 500 as a common image size
# and instructs us to reduce the dimensions by approximately 4x.
IMG_SIZE = (94, 125)


# -------------------------------------------------
# Image Loading Function
# -------------------------------------------------

def pixels_from_path(file_path):
    """
    Load an image, convert it to RGB, resize it to the assignment's
    required dimensions, and return its pixel values as a NumPy array.
    """

    with Image.open(file_path) as im:
        im = im.convert("RGB")
        im = im.resize(IMG_SIZE)

        np_im = np.array(im)

    return np_im


# -------------------------------------------------
# Verify Image Shapes
# -------------------------------------------------

print("=" * 60)
print("CSC580 - CNN Cats vs. Dogs Classification")
print("=" * 60)

print("\nChecking resized image dimensions...")

shape_counts = defaultdict(int)

cat_files = glob.glob(CAT_PATH)

for i, cat in enumerate(cat_files[:1000]):

    if i % 100 == 0:
        print(f"Processed {i} images")

    img_shape = pixels_from_path(cat).shape

    shape_counts[str(img_shape)] += 1


shape_items = list(shape_counts.items())

shape_items.sort(
    key=lambda x: x[1],
    reverse=True
)


print("\nMost common resized image shapes:")

for shape, count in shape_items[:5]:
    print(f"{shape}: {count}")


# -------------------------------------------------
# Verify Individual Image
# -------------------------------------------------

sample_shape = pixels_from_path(cat_files[5]).shape

print("\nSample image shape:", sample_shape)

print("\nStep 1A complete.")
print("=" * 60)

# -------------------------------------------------
# Step 1B - Build Training and Validation Sets
# -------------------------------------------------

# Assignment configuration
SAMPLE_SIZE = 2048
VALID_SIZE = 512

dog_files = glob.glob(DOG_PATH)

print("\n" + "=" * 60)
print("Loading Training and Validation Data")
print("=" * 60)


# -------------------------------------------------
# Load Training Images
# -------------------------------------------------

print("\nLoading training cat images...")

cat_train_set = np.asarray(
    [
        pixels_from_path(cat)
        for cat in cat_files[:SAMPLE_SIZE]
    ]
)

print("Loading training dog images...")

dog_train_set = np.asarray(
    [
        pixels_from_path(dog)
        for dog in dog_files[:SAMPLE_SIZE]
    ]
)


# -------------------------------------------------
# Load Validation Images
# -------------------------------------------------

print("Loading validation cat images...")

cat_valid_set = np.asarray(
    [
        pixels_from_path(cat)
        for cat in cat_files[-VALID_SIZE:]
    ]
)

print("Loading validation dog images...")

dog_valid_set = np.asarray(
    [
        pixels_from_path(dog)
        for dog in dog_files[-VALID_SIZE:]
    ]
)


# -------------------------------------------------
# Combine Classes
# -------------------------------------------------

x_train = np.concatenate(
    [cat_train_set, dog_train_set]
)

labels_train = np.asarray(
    [1 for _ in range(SAMPLE_SIZE)] +
    [0 for _ in range(SAMPLE_SIZE)]
)

x_valid = np.concatenate(
    [cat_valid_set, dog_valid_set]
)

labels_valid = np.asarray(
    [1 for _ in range(VALID_SIZE)] +
    [0 for _ in range(VALID_SIZE)]
)


# -------------------------------------------------
# Verify Dataset
# -------------------------------------------------

print("\nDataset Summary")
print("-" * 60)

print("Training image shape:   ", x_train.shape)
print("Training label shape:   ", labels_train.shape)

print("Validation image shape: ", x_valid.shape)
print("Validation label shape: ", labels_valid.shape)

print("\nTraining cats: ", np.sum(labels_train == 1))
print("Training dogs: ", np.sum(labels_train == 0))

print("Validation cats:", np.sum(labels_valid == 1))
print("Validation dogs:", np.sum(labels_valid == 0))

print("\nPixel value range:")
print("Minimum:", x_train.min())
print("Maximum:", x_train.max())

print("\nStep 1B complete.")
print("=" * 60)

# -------------------------------------------------
# Step 2 - Fully Connected Baseline Model
# -------------------------------------------------

print("\n" + "=" * 60)
print("STEP 2 - FULLY CONNECTED BASELINE MODEL")
print("=" * 60)


# -------------------------------------------------
# Prepare Labels for Keras 3
# -------------------------------------------------

# The original assignment stores labels as one-dimensional arrays
# with shape (n,). The model produces predictions with shape (n, 1).
# Keras 3 requires the target and output tensors to have the same rank.
#
# Reshaping changes only the array structure, not the class labels:
# Cat = 1
# Dog = 0

baseline_labels_train = labels_train.reshape(-1, 1)
baseline_labels_valid = labels_valid.reshape(-1, 1)

print("\nLabel shapes for baseline model:")
print("Training labels:  ", baseline_labels_train.shape)
print("Validation labels:", baseline_labels_valid.shape)


# -------------------------------------------------
# Build Baseline Model
# -------------------------------------------------

img_size = IMG_SIZE
fc_size = 512

# PIL uses (width, height), while NumPy represents images as
# (height, width, channels).
inputs = keras.Input(
    shape=(img_size[1], img_size[0], 3),
    name="ani_image"
)

# Flatten each image into a single vector of pixel values.
x = layers.Flatten(
    name="flattened_img"
)(inputs)

# Single fully connected hidden layer.
x = layers.Dense(
    fc_size,
    activation="relu",
    name="first_layer"
)(x)

# Binary classification output:
# 1 = cat
# 0 = dog
outputs = layers.Dense(
    1,
    activation="sigmoid",
    name="class"
)(x)

baseline_model = keras.Model(
    inputs=inputs,
    outputs=outputs
)


# -------------------------------------------------
# Display Model Architecture
# -------------------------------------------------

print("\nBaseline Model Architecture")
print("-" * 60)

baseline_model.summary()


# -------------------------------------------------
# Compile Model
# -------------------------------------------------

# The original assignment uses Adam(lr=0.001).
# Modern Keras uses the learning_rate parameter.
custom_adam = keras.optimizers.Adam(
    learning_rate=0.001
)

baseline_model.compile(
    optimizer=custom_adam,

    # Required by the assignment.
    loss="mean_squared_error",

    # The first two metrics are specified by the assignment.
    # Accuracy is included for easier interpretation of the
    # binary classification results.
    metrics=[
        "binary_crossentropy",
        "mean_squared_error",
        "accuracy"
    ]
)


# -------------------------------------------------
# Train Baseline Model
# -------------------------------------------------

print("\nTraining fully connected baseline model...")
print("-" * 60)

baseline_history = baseline_model.fit(
    x_train,
    baseline_labels_train,

    batch_size=32,

    # Important because the images were loaded with all cats
    # first and all dogs second.
    shuffle=True,

    # Written assignment instructions specify 10 epochs.
    epochs=10,

    validation_data=(
        x_valid,
        baseline_labels_valid
    )
)


# -------------------------------------------------
# Evaluate Baseline Model
# -------------------------------------------------

print("\n" + "=" * 60)
print("BASELINE MODEL RESULTS")
print("=" * 60)

baseline_results = baseline_model.evaluate(
    x_valid,
    baseline_labels_valid,
    verbose=0,
    return_dict=True
)

print("\nValidation Results")
print("-" * 60)

for metric_name, metric_value in baseline_results.items():
    print(f"{metric_name}: {metric_value:.4f}")


# -------------------------------------------------
# Final Summary
# -------------------------------------------------

print("\nBaseline Model Configuration")
print("-" * 60)

print("Input shape:       ", (img_size[1], img_size[0], 3))
print("Hidden units:      ", fc_size)
print("Optimizer:          Adam")
print("Learning rate:      0.001")
print("Loss function:      Mean Squared Error")
print("Batch size:         32")
print("Epochs:             10")
print("Shuffle:            True")
print("Training samples:   ", len(x_train))
print("Validation samples: ", len(x_valid))

print("\nStep 2 complete.")
print("=" * 60)

# -------------------------------------------------
# Step 3 - Convolutional Neural Network
# -------------------------------------------------

print("\n" + "=" * 60)
print("STEP 3 - CONVOLUTIONAL NEURAL NETWORK")
print("=" * 60)


# -------------------------------------------------
# CNN Configuration
# -------------------------------------------------

fc_layer_size = 128
img_size = IMG_SIZE


# -------------------------------------------------
# Build CNN
# -------------------------------------------------

conv_inputs = keras.Input(
    shape=(img_size[1], img_size[0], 3),
    name="cnn_image"
)

# Convolutional layer:
# 24 filters with a 3 x 3 kernel.
conv_x = layers.Conv2D(
    filters=24,
    kernel_size=3,
    activation="relu",
    name="conv_layer"
)(conv_inputs)

# Reduce spatial dimensions using max pooling.
conv_x = layers.MaxPool2D(
    pool_size=(2, 2),
    name="max_pool"
)(conv_x)

# Convert learned feature maps into a vector.
conv_x = layers.Flatten(
    name="flattened_features"
)(conv_x)

# First fully connected layer.
conv_x = layers.Dense(
    fc_layer_size,
    activation="relu",
    name="first_dense_layer"
)(conv_x)

# Second fully connected layer.
conv_x = layers.Dense(
    fc_layer_size,
    activation="relu",
    name="second_dense_layer"
)(conv_x)

# Binary classification:
# 1 = cat
# 0 = dog
conv_outputs = layers.Dense(
    1,
    activation="sigmoid",
    name="class"
)(conv_x)

conv_model = keras.Model(
    inputs=conv_inputs,
    outputs=conv_outputs
)


# -------------------------------------------------
# Display CNN Architecture
# -------------------------------------------------

print("\nCNN Model Architecture")
print("-" * 60)

conv_model.summary()


# -------------------------------------------------
# Compile CNN
# -------------------------------------------------

# Assignment specifies Adam with learning rate 1e-6.
# Modern Keras uses learning_rate rather than lr.
cnn_adam = keras.optimizers.Adam(
    learning_rate=1e-6
)

conv_model.compile(
    optimizer=cnn_adam,
    loss="binary_crossentropy",
    metrics=[
        "binary_crossentropy",
        "mean_squared_error",
        "accuracy"
    ]
)


# -------------------------------------------------
# Train CNN
# -------------------------------------------------

print("\nTraining CNN...")
print("-" * 60)

cnn_history = conv_model.fit(
    x_train,
    baseline_labels_train,

    batch_size=32,

    # Required because cats and dogs were loaded separately.
    shuffle=True,

    epochs=5,

    validation_data=(
        x_valid,
        baseline_labels_valid
    )
)


# -------------------------------------------------
# Evaluate CNN
# -------------------------------------------------

print("\n" + "=" * 60)
print("CNN VALIDATION RESULTS")
print("=" * 60)

cnn_results = conv_model.evaluate(
    x_valid,
    baseline_labels_valid,
    verbose=0,
    return_dict=True
)

for metric_name, metric_value in cnn_results.items():
    print(f"{metric_name}: {metric_value:.4f}")


# -------------------------------------------------
# Generate Validation Predictions
# -------------------------------------------------

print("\nGenerating validation predictions...")

preds = conv_model.predict(
    x_valid,
    verbose=0
)

# Convert predictions from shape (1024, 1)
# to a one-dimensional array.
preds = preds.flatten()

# Flatten validation labels for correlation calculation.
correlation_labels = baseline_labels_valid.flatten()


# -------------------------------------------------
# Pearson Correlation
# -------------------------------------------------

pearson_correlation = np.corrcoef(
    preds,
    correlation_labels
)[0, 1]


# -------------------------------------------------
# Step 3 Summary
# -------------------------------------------------

print("\n" + "=" * 60)
print("CNN MODEL SUMMARY")
print("=" * 60)

print(f"Validation Accuracy:   {cnn_results['accuracy']:.4f}")
print(f"Validation Loss:       {cnn_results['loss']:.4f}")
print(f"Validation MSE:        {cnn_results['mean_squared_error']:.4f}")
print(f"Pearson Correlation:   {pearson_correlation:.4f}")

print("\nCNN Configuration")
print("-" * 60)

print("Convolution filters: 24")
print("Kernel size:         3 x 3")
print("Pooling:             2 x 2 Max Pooling")
print("Dense layer 1:       128")
print("Dense layer 2:       128")
print("Optimizer:           Adam")
print("Learning rate:       1e-6")
print("Loss:                Binary Cross-Entropy")
print("Epochs:              5")
print("Batch size:          32")

print("\nStep 3 complete.")
print("=" * 60)

# -------------------------------------------------
# Step 4 - CNN with Second Convolutional Layer
# -------------------------------------------------

print("\n" + "=" * 60)
print("STEP 4 - TWO-LAYER CONVOLUTIONAL NEURAL NETWORK")
print("=" * 60)


# -------------------------------------------------
# Build Modified CNN
# -------------------------------------------------

deep_conv_inputs = keras.Input(
    shape=(img_size[1], img_size[0], 3),
    name="deep_cnn_image"
)

# First convolutional layer:
# Preserve the original 24-filter layer from Step 3.
deep_x = layers.Conv2D(
    filters=24,
    kernel_size=3,
    activation="relu",
    name="conv_layer_1"
)(deep_conv_inputs)

deep_x = layers.MaxPool2D(
    pool_size=(2, 2),
    name="max_pool_1"
)(deep_x)

# Second convolutional layer:
# Assignment asks us to add another layer using 48 kernels.
deep_x = layers.Conv2D(
    filters=48,
    kernel_size=3,
    activation="relu",
    name="conv_layer_2"
)(deep_x)

deep_x = layers.MaxPool2D(
    pool_size=(2, 2),
    name="max_pool_2"
)(deep_x)

# Convert feature maps into a vector.
deep_x = layers.Flatten(
    name="flattened_features"
)(deep_x)

# Two fully connected layers.
deep_x = layers.Dense(
    128,
    activation="relu",
    name="first_dense_layer"
)(deep_x)

deep_x = layers.Dense(
    128,
    activation="relu",
    name="second_dense_layer"
)(deep_x)

# Binary output:
# 1 = cat
# 0 = dog
deep_outputs = layers.Dense(
    1,
    activation="sigmoid",
    name="class"
)(deep_x)

deep_conv_model = keras.Model(
    inputs=deep_conv_inputs,
    outputs=deep_outputs
)


# -------------------------------------------------
# Display Architecture
# -------------------------------------------------

print("\nTwo-Layer CNN Architecture")
print("-" * 60)

deep_conv_model.summary()


# -------------------------------------------------
# Compile Model
# -------------------------------------------------

deep_adam = keras.optimizers.Adam(
    learning_rate=1e-6
)

deep_conv_model.compile(
    optimizer=deep_adam,
    loss="binary_crossentropy",
    metrics=[
        "binary_crossentropy",
        "mean_squared_error",
        "accuracy"
    ]
)


# -------------------------------------------------
# Train Model
# -------------------------------------------------

print("\nTraining two-layer CNN...")
print("-" * 60)

deep_history = deep_conv_model.fit(
    x_train,
    baseline_labels_train,
    batch_size=32,
    shuffle=True,
    epochs=5,
    validation_data=(
        x_valid,
        baseline_labels_valid
    )
)


# -------------------------------------------------
# Evaluate Model
# -------------------------------------------------

deep_results = deep_conv_model.evaluate(
    x_valid,
    baseline_labels_valid,
    verbose=0,
    return_dict=True
)


# -------------------------------------------------
# Generate Predictions
# -------------------------------------------------

deep_preds = deep_conv_model.predict(
    x_valid,
    verbose=0
).flatten()

correlation_labels = baseline_labels_valid.flatten()


# -------------------------------------------------
# Pearson Correlation
# -------------------------------------------------

deep_correlation = np.corrcoef(
    deep_preds,
    correlation_labels
)[0, 1]


# -------------------------------------------------
# Compare Step 3 and Step 4
# -------------------------------------------------

print("\n" + "=" * 60)
print("CNN ARCHITECTURE COMPARISON")
print("=" * 60)

print("\nStep 3 - One Convolutional Layer")
print("-" * 60)
print(f"Validation Accuracy: {cnn_results['accuracy']:.4f}")
print(f"Validation MSE:      {cnn_results['mean_squared_error']:.4f}")
print(f"Pearson Correlation: {pearson_correlation:.4f}")

print("\nStep 4 - Two Convolutional Layers")
print("-" * 60)
print(f"Validation Accuracy: {deep_results['accuracy']:.4f}")
print(f"Validation MSE:      {deep_results['mean_squared_error']:.4f}")
print(f"Pearson Correlation: {deep_correlation:.4f}")


# -------------------------------------------------
# Calculate Change
# -------------------------------------------------

accuracy_change = (
    deep_results["accuracy"] -
    cnn_results["accuracy"]
)

correlation_change = (
    deep_correlation -
    pearson_correlation
)

print("\nChange in Performance")
print("-" * 60)

print(
    f"Accuracy change:     "
    f"{accuracy_change:+.4f}"
)

print(
    f"Correlation change:  "
    f"{correlation_change:+.4f}"
)

print("\nStep 4A complete.")
print("=" * 60)

# -------------------------------------------------
# Step 4B - Prediction and Threshold Analysis
# -------------------------------------------------

print("\n" + "=" * 60)
print("STEP 4B - PREDICTION THRESHOLD ANALYSIS")
print("=" * 60)


# -------------------------------------------------
# Standard Classification Accuracy
# -------------------------------------------------

# Convert predicted probabilities into binary classifications.
# Probability >= 0.50 = Cat
# Probability <  0.50 = Dog

predicted_classes = (deep_preds >= 0.50).astype(int)

classification_accuracy = np.mean(
    predicted_classes == correlation_labels
)

print("\nStandard Classification Results")
print("-" * 60)

print(f"Classification threshold: 0.50")
print(f"Classification accuracy:  {classification_accuracy:.4f}")
print(
    f"Classification accuracy:  "
    f"{classification_accuracy * 100:.2f}%"
)


# -------------------------------------------------
# Threshold Analysis
# -------------------------------------------------

print("\nCat Probability Threshold Analysis")
print("-" * 60)

print(
    f"{'Threshold':<12}"
    f"{'Images':<12}"
    f"{'Actual Cats':<15}"
    f"{'Cat Proportion':<15}"
)

threshold_results = []

for i in range(1, 10):

    threshold = i / 10

    # Select images for which the model predicts a cat
    # probability greater than the current threshold.
    mask = deep_preds > threshold

    image_count = np.sum(mask)

    if image_count > 0:

        actual_cats = np.sum(
            correlation_labels[mask]
        )

        cat_proportion = (
            actual_cats / image_count
        )

    else:
        actual_cats = 0
        cat_proportion = np.nan

    threshold_results.append(
        (
            threshold,
            image_count,
            actual_cats,
            cat_proportion
        )
    )

    print(
        f"{threshold:<12.1f}"
        f"{image_count:<12}"
        f"{actual_cats:<15}"
        f"{cat_proportion:<15.4f}"
    )


# -------------------------------------------------
# Scatterplot
# -------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))

sns.scatterplot(
    x=deep_preds,
    y=correlation_labels,
    alpha=0.5
)

plt.xlabel("Predicted Probability of Cat")
plt.ylabel("Actual Class (0 = Dog, 1 = Cat)")

plt.title(
    "CNN Predicted Cat Probability vs. Actual Class"
)

plt.yticks(
    [0, 1],
    ["Dog (0)", "Cat (1)"]
)

plt.tight_layout()

plt.savefig(
    "cnn_prediction_scatterplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# -------------------------------------------------
# Summary
# -------------------------------------------------

print("\n" + "=" * 60)
print("STEP 4B SUMMARY")
print("=" * 60)

print(
    f"Validation Accuracy:    "
    f"{classification_accuracy:.4f}"
)

print(
    f"Pearson Correlation:    "
    f"{deep_correlation:.4f}"
)

print(
    "Scatterplot saved as:  "
    "cnn_prediction_scatterplot.png"
)

print("\nStep 4B complete.")
print("=" * 60)

# ============================================================
# STEP 5 - INDIVIDUAL IMAGE PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("STEP 5 - INDIVIDUAL IMAGE PREDICTION")
print("=" * 60)


def animal_pic(index):
    """
    Returns an image from the validation dataset.
    """
    return Image.fromarray(x_valid[index])


def cat_probability(index):
    """
    Predicts the probability that a validation image is a cat.
    """

    image = np.asarray([x_valid[index]])

    prediction = deep_conv_model.predict(
        image,
        verbose=0
    )[0][0]

    return float(prediction)


def actual_class(index):
    """
    Returns the actual class label for a validation image.
    1 = Cat
    0 = Dog
    """

    return "Cat" if labels_valid[index] == 1 else "Dog"


def predicted_class(probability, threshold=0.50):
    """
    Converts the predicted probability into a class.
    """

    return "Cat" if probability >= threshold else "Dog"


# Test the professor's example index
example_index = 600

probability = cat_probability(example_index)

print(f"\nExample validation image index: {example_index}")
print(f"Actual class:                  {actual_class(example_index)}")
print(f"Probability of being a cat:   {probability:.4f}")
print(f"Predicted class:               {predicted_class(probability)}")

print("\nStep 5 complete.")
print("=" * 60)


# ============================================================
# STEP 6 - SAVE FINAL CNN MODEL
# ============================================================

print("\n" + "=" * 60)
print("STEP 6 - SAVE FINAL CNN MODEL")
print("=" * 60)

model_path = "conv_model_big.keras"

deep_conv_model.save(model_path)

print(f"\nModel saved successfully: {model_path}")

print("\nStep 6 complete.")
print("=" * 60)


# ============================================================
# STEP 7 - USER PREDICTION INTERFACE
# ============================================================

print("\n" + "=" * 60)
print("STEP 7 - CAT VS. DOG PREDICTION INTERFACE")
print("=" * 60)

print(
    f"\nThe validation dataset contains "
    f"{len(x_valid)} images."
)

print(
    f"Choose an image index from "
    f"0 to {len(x_valid) - 1}."
)


def prediction_interface():
    """
    Allows the user to select a validation image by index
    and displays the CNN's prediction.
    """

    while True:

        user_input = input(
            "\nEnter an image index "
            "(or q to quit): "
        )

        if user_input.lower() == "q":
            print("\nPrediction interface closed.")
            break

        try:
            index = int(user_input)

            # Validate index
            if index < 0 or index >= len(x_valid):
                print(
                    f"Please enter an index between "
                    f"0 and {len(x_valid) - 1}."
                )
                continue

            # Generate prediction
            probability = cat_probability(index)

            actual = actual_class(index)

            prediction = predicted_class(probability)

            correct = actual == prediction

            # Print results
            print("\n" + "-" * 60)
            print("CNN IMAGE CLASSIFICATION RESULT")
            print("-" * 60)

            print(f"Image index:                {index}")
            print(f"Actual class:               {actual}")
            print(f"Predicted class:            {prediction}")

            print(
                f"Probability of being a cat: "
                f"{probability:.4f}"
            )

            print(
                f"Probability of being a dog: "
                f"{1 - probability:.4f}"
            )

            print(
                f"Classification correct:     "
                f"{'Yes' if correct else 'No'}"
            )

            print("-" * 60)

            # Display selected validation image
            plt.figure(figsize=(6, 6))

            plt.imshow(x_valid[index])

            plt.axis("off")

            plt.title(
                f"Actual: {actual} | "
                f"Predicted: {prediction}\n"
                f"Cat Probability: {probability:.4f}"
            )

            plt.tight_layout()

            plt.show()

        except ValueError:

            print(
                "Invalid input. Enter a numerical "
                "image index or q to quit."
            )


# Start interface
prediction_interface()

print("\nStep 7 complete.")
print("=" * 60)