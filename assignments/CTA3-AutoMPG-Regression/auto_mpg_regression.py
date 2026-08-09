"""
CSC580 - CTA3
TensorFlow Regression using the Auto MPG Dataset

Author: Christine DeLuna

Objective:
Build and train a TensorFlow regression model to predict vehicle fuel
efficiency (MPG) using the Auto MPG dataset.
"""

# =====================================================
# Import Required Libraries
# =====================================================

from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

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
    "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
)

print("Dataset downloaded to:")
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

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

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

example_result = model.predict(example_batch)

print("\nPredictions from the Untrained Model:")
print(example_result)

# =====================================================
# Step 14 - Train the Neural Network
# =====================================================

EPOCHS = 1000

history = model.fit(
    normed_train_data,
    train_labels,
    epochs=EPOCHS,
    validation_split=0.2,
    verbose=0,
    callbacks=[tfdocs.modeling.EpochDots()]
)

# =====================================================
# Step 15 - Review the Training History
# =====================================================

hist = pd.DataFrame(history.history)
hist["epoch"] = history.epoch

print("\nTraining History:")
print(hist.tail())

# =====================================================
# Step 16 - Plot Mean Absolute Error (MAE)
# =====================================================

plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

plotter.plot({"Basic": history}, metric="mae")

plt.ylim([0, 10])

plt.ylabel("MAE [MPG]")

plt.show()

# =====================================================
# Step 17 - Plot Mean Squared Error (MSE)
# =====================================================

plotter.plot({"Basic": history}, metric="mse")

plt.ylim([0, 20])

plt.ylabel("MSE [MPG²]")

plt.show()