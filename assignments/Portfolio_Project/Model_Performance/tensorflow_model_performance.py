"""
CSC580 - Portfolio Project Milestone
Improving TensorFlow Model Performance and Quality

Author: Christine DeLuna

Objective:
Improve the TensorFlow regression model developed in Module 3 by
implementing EarlyStopping and evaluating model performance using
the Auto MPG testing dataset.
"""

# =====================================================
# Import Required Libraries
# =====================================================

from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

# Display TensorFlow version
print("TensorFlow Version:", tf.__version__)


# =====================================================
# Step 1 - Download the Auto MPG Dataset
# =====================================================

dataset_path = keras.utils.get_file(
    "auto-mpg.data",
    "http://archive.ics.uci.edu/ml/machine-learning-databases/"
    "auto-mpg/auto-mpg.data"
)

print("\nDataset downloaded to:")
print(dataset_path)


# =====================================================
# Step 2 - Import the Dataset Using Pandas
# =====================================================

column_names = [
    "MPG",
    "Cylinders",
    "Displacement",
    "Horsepower",
    "Weight",
    "Acceleration",
    "Model Year",
    "Origin"
]

raw_dataset = pd.read_csv(
    dataset_path,
    names=column_names,
    na_values="?",
    comment="\t",
    sep=" ",
    skipinitialspace=True
)

# Make a copy so we preserve the original data
dataset = raw_dataset.copy()


# =====================================================
# Step 3 - Display the Last Five Rows
# =====================================================

print("\nDataset Tail:")
print(dataset.tail())


# =====================================================
# Step 4 - Check for Missing Values
# =====================================================

print("\nMissing Values:")
print(dataset.isna().sum())


# =====================================================
# Step 5 - Remove Missing Values
# =====================================================

dataset = dataset.dropna()

print("\nDataset Shape After Cleaning:")
print(dataset.shape)


# =====================================================
# Step 6 - Split into Training and Testing Data
# =====================================================

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

print("\nTraining Dataset Shape:", train_dataset.shape)
print("Testing Dataset Shape:", test_dataset.shape)


# =====================================================
# Step 7 - Inspect the Data
# =====================================================

sns.pairplot(
    train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]],
    diag_kind="kde"
)

plt.show()


# =====================================================
# Step 8 - Review Dataset Statistics
# =====================================================

train_stats = train_dataset.describe()

# Remove the target variable (MPG)
train_stats.pop("MPG")

# Transpose the table for easier reading
train_stats = train_stats.transpose()

print("\nTraining Statistics:")
print(train_stats)


# =====================================================
# Step 9 - Separate Features from Labels
# =====================================================

train_labels = train_dataset.pop("MPG")
test_labels = test_dataset.pop("MPG")

print("\nTraining Features Shape:", train_dataset.shape)
print("Training Labels Shape:", train_labels.shape)

print("\nTesting Features Shape:", test_dataset.shape)
print("Testing Labels Shape:", test_labels.shape)


# =====================================================
# Step 10 - Normalize the Data
# =====================================================

def norm(x):
    return (x - train_stats["mean"]) / train_stats["std"]


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

print("\nNormalized Training Data:")
print(normed_train_data.head())


# =====================================================
# Step 11 - Build the Neural Network
# =====================================================

def build_model():

    model = keras.Sequential([
        layers.Input(shape=(len(train_dataset.keys()),)),
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=0.001
    )

    model.compile(
        loss="mse",
        optimizer=optimizer,
        metrics=["mae", "mse"]
    )

    return model


model = build_model()


# =====================================================
# Step 12 - Inspect the Model
# =====================================================

print("\nModel Summary:")
model.summary()


# =====================================================
# Step 13 - Test the Untrained Model
# =====================================================

example_batch = normed_train_data[:10]

example_result = model.predict(
    example_batch,
    verbose=0
)

print("\nPredictions from the Untrained Model:")
print(example_result)


# =====================================================
# Step 14 - Configure Early Stopping
# =====================================================

EPOCHS = 1000

# Stop training if validation loss does not improve
# for 10 consecutive epochs.
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)


# =====================================================
# Step 15 - Train the Neural Network
# =====================================================

early_history = model.fit(
    normed_train_data,
    train_labels,
    epochs=EPOCHS,
    validation_split=0.2,
    verbose=0,
    callbacks=[
        early_stop,
        tfdocs.modeling.EpochDots()
    ]
)

print(
    "\nTraining stopped after",
    len(early_history.epoch),
    "epochs."
)


# =====================================================
# Step 16 - Review the Training History
# =====================================================

hist = pd.DataFrame(early_history.history)
hist["epoch"] = early_history.epoch

print("\nTraining History:")
print(hist.tail())


# =====================================================
# Step 17 - Plot Mean Absolute Error with Early Stopping
# =====================================================

plotter = tfdocs.plots.HistoryPlotter(
    smoothing_std=2
)

plotter.plot(
    {"Early Stopping": early_history},
    metric="mae"
)

plt.ylim([0, 10])
plt.ylabel("MAE [MPG]")
plt.title("Model Mean Absolute Error with Early Stopping")

plt.show()


# =====================================================
# Step 18 - Evaluate the Model Using the Test Set
# =====================================================

loss, mae, mse = model.evaluate(
    normed_test_data,
    test_labels,
    verbose=2
)

print(
    "\nTesting Set Mean Abs Error: {:5.2f} MPG".format(mae)
)

print(
    "Testing Set Mean Squared Error: {:5.2f}".format(mse)
)


# =====================================================
# Step 19 - Make Predictions Using the Test Set
# =====================================================

test_predictions = model.predict(
    normed_test_data,
    verbose=0
).flatten()

print("\nSample Test Predictions:")

for true_value, predicted_value in zip(
    test_labels[:10],
    test_predictions[:10]
):
    print(
        f"True MPG: {true_value:5.1f} | "
        f"Predicted MPG: {predicted_value:5.1f}"
    )


# =====================================================
# Step 20 - Plot True MPG vs. Predicted MPG
# =====================================================

plt.figure()

plt.axes(aspect="equal")

plt.scatter(
    test_labels,
    test_predictions
)

plt.xlabel("True Values [MPG]")
plt.ylabel("Predictions [MPG]")
plt.title("True MPG vs. Predicted MPG")

lims = [0, 50]

plt.xlim(lims)
plt.ylim(lims)

plt.plot(lims, lims)

plt.show()


# =====================================================
# Step 21 - Calculate Prediction Error
# =====================================================

error = test_predictions - test_labels

print("\nPrediction Error Statistics:")
print(error.describe())


# =====================================================
# Step 22 - Plot Prediction Error Distribution
# =====================================================

plt.figure()

plt.hist(
    error,
    bins=25
)

plt.xlabel("Prediction Error [MPG]")
plt.ylabel("Count")
plt.title("Distribution of Prediction Errors")

plt.show()