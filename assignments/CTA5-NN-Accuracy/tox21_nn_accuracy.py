"""
CSC580 - Critical Thinking Assignment
Improving the Accuracy of a Neural Network

Author: Christine DeLuna

Objective:
Evaluate and improve neural network performance on the DeepChem Tox21
dataset through hyperparameter optimization.

This implementation builds upon the Tox21 classification work completed
in Module 4. The original assignment materials contain legacy TensorFlow
1.x syntax. Where necessary, the implementation will be modernized for
TensorFlow 2.21 and the current Keras API while preserving the intended
experimental design.
"""

# =====================================================
# Import Required Libraries
# =====================================================

import numpy as np
import matplotlib.pyplot as plt
import deepchem as dc

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# =====================================================
# Reproducibility
# =====================================================

SEED = 456

np.random.seed(SEED)
tf.random.set_seed(SEED)


# =====================================================
# Display Library Versions
# =====================================================

print("=" * 60)
print("CSC580 - Tox21 Neural Network Accuracy Optimization")
print("=" * 60)

print("\nTensorFlow Version:", tf.__version__)
print("DeepChem Version:", dc.__version__)


# =====================================================
# Step 1 - Load the Tox21 Dataset
# =====================================================

print("\nLoading Tox21 dataset...")

_, (train, valid, test), _ = dc.molnet.load_tox21()


# =====================================================
# Extract Features, Labels, and Weights
# =====================================================

train_X, train_y, train_w = train.X, train.y, train.w
valid_X, valid_y, valid_w = valid.X, valid.y, valid.w
test_X, test_y, test_w = test.X, test.y, test.w


# =====================================================
# Remove Extra Tox21 Tasks
# =====================================================
#
# Tox21 contains multiple prediction tasks.
# The assignment evaluates only the first task.
# =====================================================

train_y = train_y[:, 0]
valid_y = valid_y[:, 0]
test_y = test_y[:, 0]

train_w = train_w[:, 0]
valid_w = valid_w[:, 0]
test_w = test_w[:, 0]


# =====================================================
# Inspect Dataset
# =====================================================

print("\nDataset Shapes")

print("Training features:  ", train_X.shape)
print("Training labels:    ", train_y.shape)

print("Validation features:", valid_X.shape)
print("Validation labels:  ", valid_y.shape)

print("Testing features:   ", test_X.shape)
print("Testing labels:     ", test_y.shape)


# =====================================================
# Step 2 - Random Forest Baseline
# =====================================================
#
# The assignment specifies a RandomForestClassifier
# containing 50 estimators with balanced class weights.
#
# This provides a non-neural-network baseline against
# which the optimized neural network can later be
# compared.
# =====================================================

print("\n" + "=" * 60)
print("Random Forest Baseline")
print("=" * 60)

random_forest = RandomForestClassifier(
    class_weight="balanced",
    n_estimators=50,
    random_state=SEED
)

print("\nAbout to fit model on training set...")

random_forest.fit(
    train_X,
    train_y
)


# =====================================================
# Generate Predictions
# =====================================================

train_y_pred = random_forest.predict(train_X)
valid_y_pred = random_forest.predict(valid_X)
test_y_pred = random_forest.predict(test_X)


# =====================================================
# Calculate Weighted Classification Accuracy
# =====================================================

train_score = accuracy_score(
    train_y,
    train_y_pred,
    sample_weight=train_w
)

valid_score = accuracy_score(
    valid_y,
    valid_y_pred,
    sample_weight=valid_w
)

test_score = accuracy_score(
    test_y,
    test_y_pred,
    sample_weight=test_w
)


# =====================================================
# Display Random Forest Results
# =====================================================

print("\nRandom Forest Results")

print(
    "Weighted Train Classification Accuracy: "
    f"{train_score:.4f}"
)

print(
    "Weighted Validation Classification Accuracy: "
    f"{valid_score:.4f}"
)

print(
    "Weighted Test Classification Accuracy: "
    f"{test_score:.4f}"
)

print("\nRandom Forest baseline complete.")

# =====================================================
# Step 3 - Configurable Tox21 Neural Network
# =====================================================
#
# The assignment-provided implementation uses legacy
# TensorFlow 1.x graph/session APIs including:
#
#   tf.Graph()
#   tf.placeholder()
#   tf.Session()
#   tf.train.AdamOptimizer()
#
# Based on compatibility issues identified during CTA4,
# this implementation preserves the requested network
# architecture and hyperparameters using the current
# TensorFlow 2.21 / Keras API.
# =====================================================


def eval_tox21_hyperparams(
        n_hidden=50,
        n_layers=1,
        learning_rate=0.001,
        dropout_prob=0.5,
        n_epochs=45,
        batch_size=100,
        weight_positives=True,
        seed=456
):

    # -------------------------------------------------
    # Display Hyperparameters
    # -------------------------------------------------

    print("\n" + "=" * 60)
    print("Tox21 Neural Network Hyperparameter Evaluation")
    print("=" * 60)

    print("n_hidden       =", n_hidden)
    print("n_layers       =", n_layers)
    print("learning_rate  =", learning_rate)
    print("dropout_prob   =", dropout_prob)
    print("n_epochs       =", n_epochs)
    print("batch_size     =", batch_size)
    print("weight_positives =", weight_positives)
    print("random_seed    =", seed)

    print("=" * 60)

    # -------------------------------------------------
    # Set Random Seeds
    # -------------------------------------------------

    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Clear previous Keras model state before each run
    keras.backend.clear_session()

    # -------------------------------------------------
    # Build Neural Network
    # -------------------------------------------------

    model = keras.Sequential()

    model.add(
        layers.Input(shape=(train_X.shape[1],))
    )

    # Add requested number of hidden layers
    for _ in range(n_layers):

        model.add(
            layers.Dense(
                n_hidden,
                activation="relu"
            )
        )

        # Modern Keras dropout uses the fraction of
        # neurons to drop rather than TF1 keep_prob.
        model.add(
            layers.Dropout(dropout_prob)
        )

    # Binary classification output
    model.add(
        layers.Dense(
            1,
            activation="sigmoid"
        )
    )

    # -------------------------------------------------
    # Configure Optimizer
    # -------------------------------------------------

    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    # -------------------------------------------------
    # Compile Model
    # -------------------------------------------------

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # -------------------------------------------------
    # Configure Sample Weights
    # -------------------------------------------------

    if weight_positives:
        training_weights = train_w
    else:
        training_weights = None

    # -------------------------------------------------
    # Train Model
    # -------------------------------------------------

    print("\nTraining neural network...")

    history = model.fit(
        train_X,
        train_y,
        sample_weight=training_weights,
        validation_data=(valid_X, valid_y, valid_w),
        epochs=n_epochs,
        batch_size=batch_size,
        verbose=0
    )

    # -------------------------------------------------
    # Generate Validation Predictions
    # -------------------------------------------------

    valid_probabilities = model.predict(
        valid_X,
        verbose=0
    ).flatten()

    valid_y_pred = (
        valid_probabilities >= 0.5
    ).astype(int)

    # -------------------------------------------------
    # Calculate Weighted Validation Accuracy
    # -------------------------------------------------

    weighted_score = accuracy_score(
        valid_y,
        valid_y_pred,
        sample_weight=valid_w
    )

    print(
        "\nValid Weighted Classification Accuracy: "
        f"{weighted_score:.4f}"
    )

    return weighted_score, model, history

# =====================================================
# Step 4 - Evaluate Default Hyperparameters
# =====================================================

baseline_nn_score, baseline_nn_model, baseline_nn_history = (
    eval_tox21_hyperparams(
        n_hidden=50,
        n_layers=1,
        learning_rate=0.001,
        dropout_prob=0.5,
        n_epochs=45,
        batch_size=100,
        weight_positives=True,
        seed=456
    )
)

# =====================================================
# Step 5 - Hyperparameter Optimization
# =====================================================
#
# The assignment asks that multiple combinations of
# hyperparameters be evaluated and that each combination
# be repeated using different random seeds.
#
# Averaging validation accuracy across multiple seeds
# reduces the likelihood that a configuration appears
# superior simply because of favorable random
# initialization.
# =====================================================

import itertools

print("\n" + "=" * 60)
print("Beginning Hyperparameter Optimization")
print("=" * 60)


# =====================================================
# Define Hyperparameter Search Space
# =====================================================

n_hidden_values = [50, 100]
n_layers_values = [1, 2]
learning_rate_values = [0.001, 0.0005]
dropout_values = [0.3, 0.5]

# Hold these parameters constant during the initial search
SEARCH_EPOCHS = 45
SEARCH_BATCH_SIZE = 100
SEARCH_WEIGHT_POSITIVES = True

# Multiple seeds reduce sensitivity to random initialization
seeds = [123, 456, 789]


# =====================================================
# Generate Hyperparameter Combinations
# =====================================================

hyperparameter_combinations = list(
    itertools.product(
        n_hidden_values,
        n_layers_values,
        learning_rate_values,
        dropout_values
    )
)

print(
    f"\nTotal hyperparameter configurations: "
    f"{len(hyperparameter_combinations)}"
)

print(
    f"Runs per configuration: {len(seeds)}"
)

print(
    f"Total neural network training runs: "
    f"{len(hyperparameter_combinations) * len(seeds)}"
)


# =====================================================
# Store Results
# =====================================================

search_results = []

best_average_score = -1
best_configuration = None


# =====================================================
# Evaluate Each Hyperparameter Combination
# =====================================================

for config_number, combination in enumerate(
        hyperparameter_combinations,
        start=1
):

    n_hidden, n_layers, learning_rate, dropout_prob = combination

    print("\n" + "#" * 60)
    print(
        f"Configuration {config_number} "
        f"of {len(hyperparameter_combinations)}"
    )
    print("#" * 60)

    print(
        f"Hidden Units: {n_hidden} | "
        f"Layers: {n_layers} | "
        f"Learning Rate: {learning_rate} | "
        f"Dropout: {dropout_prob}"
    )

    seed_scores = []

    # -------------------------------------------------
    # Repeat Configuration Across Random Seeds
    # -------------------------------------------------

    for seed in seeds:

        score, _, _ = eval_tox21_hyperparams(
            n_hidden=n_hidden,
            n_layers=n_layers,
            learning_rate=learning_rate,
            dropout_prob=dropout_prob,
            n_epochs=SEARCH_EPOCHS,
            batch_size=SEARCH_BATCH_SIZE,
            weight_positives=SEARCH_WEIGHT_POSITIVES,
            seed=seed
        )

        seed_scores.append(score)

    # -------------------------------------------------
    # Average Performance Across Seeds
    # -------------------------------------------------

    average_score = np.mean(seed_scores)
    score_std = np.std(seed_scores)

    print("\nConfiguration Summary")
    print("-" * 60)

    print(
        "Seed Scores:",
        [f"{score:.4f}" for score in seed_scores]
    )

    print(
        f"Average Validation Accuracy: "
        f"{average_score:.4f}"
    )

    print(
        f"Standard Deviation: "
        f"{score_std:.4f}"
    )

    # -------------------------------------------------
    # Store Configuration Results
    # -------------------------------------------------

    result = {
        "n_hidden": n_hidden,
        "n_layers": n_layers,
        "learning_rate": learning_rate,
        "dropout_prob": dropout_prob,
        "n_epochs": SEARCH_EPOCHS,
        "batch_size": SEARCH_BATCH_SIZE,
        "weight_positives": SEARCH_WEIGHT_POSITIVES,
        "seed_1_score": seed_scores[0],
        "seed_2_score": seed_scores[1],
        "seed_3_score": seed_scores[2],
        "average_score": average_score,
        "std_score": score_std
    }

    search_results.append(result)

    # -------------------------------------------------
    # Track Best Configuration
    # -------------------------------------------------

    if average_score > best_average_score:

        best_average_score = average_score
        best_configuration = result


# =====================================================
# Display Best Initial Configuration
# =====================================================

print("\n" + "=" * 60)
print("BEST HYPERPARAMETER CONFIGURATION")
print("=" * 60)

print(
    f"Hidden Units: "
    f"{best_configuration['n_hidden']}"
)

print(
    f"Hidden Layers: "
    f"{best_configuration['n_layers']}"
)

print(
    f"Learning Rate: "
    f"{best_configuration['learning_rate']}"
)

print(
    f"Dropout: "
    f"{best_configuration['dropout_prob']}"
)

print(
    f"Epochs: "
    f"{best_configuration['n_epochs']}"
)

print(
    f"Batch Size: "
    f"{best_configuration['batch_size']}"
)

print(
    f"Average Validation Accuracy: "
    f"{best_configuration['average_score']:.4f}"
)

print(
    f"Standard Deviation: "
    f"{best_configuration['std_score']:.4f}"
)

print("=" * 60)

# =====================================================
# Step 6 - Refine the Best Hyperparameter Configuration
# =====================================================
#
# The initial search identified the strongest architecture
# as:
#
#   Hidden units: 50
#   Hidden layers: 2
#   Learning rate: 0.001
#   Dropout: 0.50
#
# These parameters are held constant while the remaining
# training hyperparameters are evaluated:
#
#   - Number of epochs
#   - Batch size
#   - Positive-class weighting
#
# Each configuration is evaluated using three random seeds
# to reduce sensitivity to random initialization.
# =====================================================

print("\n" + "=" * 60)
print("Beginning Hyperparameter Refinement")
print("=" * 60)


# =====================================================
# Best Architecture from Initial Search
# =====================================================

BEST_HIDDEN = 50
BEST_LAYERS = 2
BEST_LEARNING_RATE = 0.001
BEST_DROPOUT = 0.5


# =====================================================
# Define Refinement Search Space
# =====================================================

epoch_values = [30, 45, 60]
batch_size_values = [50, 100]
weight_positive_values = [True, False]

refinement_seeds = [123, 456, 789]


# =====================================================
# Generate Refinement Combinations
# =====================================================

refinement_combinations = list(
    itertools.product(
        epoch_values,
        batch_size_values,
        weight_positive_values
    )
)

print(
    f"\nTotal refinement configurations: "
    f"{len(refinement_combinations)}"
)

print(
    f"Runs per configuration: "
    f"{len(refinement_seeds)}"
)

print(
    f"Total refinement training runs: "
    f"{len(refinement_combinations) * len(refinement_seeds)}"
)


# =====================================================
# Store Refinement Results
# =====================================================

refinement_results = []

best_refinement_score = -1
best_refinement_configuration = None


# =====================================================
# Evaluate Refinement Combinations
# =====================================================

for config_number, combination in enumerate(
        refinement_combinations,
        start=1
):

    n_epochs, batch_size, weight_positives = combination

    print("\n" + "#" * 60)

    print(
        f"Refinement Configuration {config_number} "
        f"of {len(refinement_combinations)}"
    )

    print("#" * 60)

    print(
        f"Epochs: {n_epochs} | "
        f"Batch Size: {batch_size} | "
        f"Weight Positives: {weight_positives}"
    )

    seed_scores = []


    # -------------------------------------------------
    # Repeat Configuration Across Random Seeds
    # -------------------------------------------------

    for seed in refinement_seeds:

        score, _, _ = eval_tox21_hyperparams(
            n_hidden=BEST_HIDDEN,
            n_layers=BEST_LAYERS,
            learning_rate=BEST_LEARNING_RATE,
            dropout_prob=BEST_DROPOUT,
            n_epochs=n_epochs,
            batch_size=batch_size,
            weight_positives=weight_positives,
            seed=seed
        )

        seed_scores.append(score)


    # -------------------------------------------------
    # Calculate Mean and Standard Deviation
    # -------------------------------------------------

    average_score = np.mean(seed_scores)
    score_std = np.std(seed_scores)

    print("\nRefinement Configuration Summary")
    print("-" * 60)

    print(
        "Seed Scores:",
        [f"{score:.4f}" for score in seed_scores]
    )

    print(
        f"Average Validation Accuracy: "
        f"{average_score:.4f}"
    )

    print(
        f"Standard Deviation: "
        f"{score_std:.4f}"
    )


    # -------------------------------------------------
    # Store Results
    # -------------------------------------------------

    result = {
        "n_hidden": BEST_HIDDEN,
        "n_layers": BEST_LAYERS,
        "learning_rate": BEST_LEARNING_RATE,
        "dropout_prob": BEST_DROPOUT,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "weight_positives": weight_positives,
        "seed_1_score": seed_scores[0],
        "seed_2_score": seed_scores[1],
        "seed_3_score": seed_scores[2],
        "average_score": average_score,
        "std_score": score_std
    }

    refinement_results.append(result)


    # -------------------------------------------------
    # Track Best Refinement Configuration
    # -------------------------------------------------

    if average_score > best_refinement_score:

        best_refinement_score = average_score
        best_refinement_configuration = result


# =====================================================
# Display Best Refined Configuration
# =====================================================

print("\n" + "=" * 60)
print("BEST REFINED HYPERPARAMETER CONFIGURATION")
print("=" * 60)

print(
    f"Hidden Units: "
    f"{best_refinement_configuration['n_hidden']}"
)

print(
    f"Hidden Layers: "
    f"{best_refinement_configuration['n_layers']}"
)

print(
    f"Learning Rate: "
    f"{best_refinement_configuration['learning_rate']}"
)

print(
    f"Dropout: "
    f"{best_refinement_configuration['dropout_prob']}"
)

print(
    f"Epochs: "
    f"{best_refinement_configuration['n_epochs']}"
)

print(
    f"Batch Size: "
    f"{best_refinement_configuration['batch_size']}"
)

print(
    f"Weight Positives: "
    f"{best_refinement_configuration['weight_positives']}"
)

print(
    f"Average Validation Accuracy: "
    f"{best_refinement_configuration['average_score']:.4f}"
)

print(
    f"Standard Deviation: "
    f"{best_refinement_configuration['std_score']:.4f}"
)

print("=" * 60)

# =====================================================
# Step 7 - Final Model Evaluation on Test Set
# =====================================================
#
# After hyperparameter selection is complete, train the
# final model using the best refined configuration and
# evaluate it against the previously unseen test set.
#
# The test set was not used to select hyperparameters,
# providing a final estimate of model generalization.
# =====================================================

print("\n" + "=" * 60)
print("FINAL MODEL TEST EVALUATION")
print("=" * 60)

# Use a fixed seed for the final reproducible model
FINAL_SEED = 456

np.random.seed(FINAL_SEED)
tf.random.set_seed(FINAL_SEED)
keras.backend.clear_session()


# =====================================================
# Build Final Optimized Model
# =====================================================

final_model = keras.Sequential([
    layers.Input(shape=(train_X.shape[1],)),

    layers.Dense(
        50,
        activation="relu"
    ),

    layers.Dropout(0.5),

    layers.Dense(
        50,
        activation="relu"
    ),

    layers.Dropout(0.5),

    layers.Dense(
        1,
        activation="sigmoid"
    )
])


final_model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# =====================================================
# Train Final Model
# =====================================================

print("\nTraining final optimized model...")

final_history = final_model.fit(
    train_X,
    train_y,
    sample_weight=train_w,
    epochs=60,
    batch_size=50,
    verbose=0
)


# =====================================================
# Generate Predictions
# =====================================================

train_probabilities = final_model.predict(
    train_X,
    verbose=0
).flatten()

valid_probabilities = final_model.predict(
    valid_X,
    verbose=0
).flatten()

test_probabilities = final_model.predict(
    test_X,
    verbose=0
).flatten()


train_predictions = (
    train_probabilities >= 0.5
).astype(int)

valid_predictions = (
    valid_probabilities >= 0.5
).astype(int)

test_predictions = (
    test_probabilities >= 0.5
).astype(int)


# =====================================================
# Calculate Weighted Classification Accuracy
# =====================================================

final_train_accuracy = accuracy_score(
    train_y,
    train_predictions,
    sample_weight=train_w
)

final_valid_accuracy = accuracy_score(
    valid_y,
    valid_predictions,
    sample_weight=valid_w
)

final_test_accuracy = accuracy_score(
    test_y,
    test_predictions,
    sample_weight=test_w
)


# =====================================================
# Display Final Results
# =====================================================

print("\nFinal Optimized Neural Network Results")
print("-" * 60)

print(
    f"Weighted Training Accuracy:   "
    f"{final_train_accuracy:.4f}"
)

print(
    f"Weighted Validation Accuracy: "
    f"{final_valid_accuracy:.4f}"
)

print(
    f"Weighted Test Accuracy:       "
    f"{final_test_accuracy:.4f}"
)

print("-" * 60)

print("\nComparison with Previous Results")

print(
    f"Baseline Neural Network Validation Accuracy: "
    f"0.6243"
)

print(
    f"Optimized NN Average Validation Accuracy:    "
    f"0.6613"
)

print(
    f"Random Forest Validation Accuracy:           "
    f"0.7232"
)

print("=" * 60)