# =====================================================
# CSC580 - CTA4
# Toxicology Classification using TensorFlow
# Christine DeLuna
# =====================================================

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import deepchem as dc

from sklearn.metrics import accuracy_score

# -----------------------------------------------------
# Make results reproducible
# -----------------------------------------------------

np.random.seed(456)
tf.random.set_seed(456)

print("TensorFlow Version:", tf.__version__)
print("DeepChem Version:", dc.__version__)

# =====================================================
# Step 1 - Load the Tox21 Dataset
# =====================================================

print("\nLoading the Tox21 dataset...")

tasks, datasets, transformers = dc.molnet.load_tox21()

train_dataset, valid_dataset, test_dataset = datasets

print("Dataset loaded successfully!")

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(valid_dataset)}")
print(f"Testing samples: {len(test_dataset)}")

# =====================================================
# Step 2 - Extract Features, Labels, and Weights
# =====================================================

train_X, train_y, train_w = train_dataset.X, train_dataset.y, train_dataset.w
valid_X, valid_y, valid_w = valid_dataset.X, valid_dataset.y, valid_dataset.w
test_X, test_y, test_w = test_dataset.X, test_dataset.y, test_dataset.w

# =====================================================
# Step 3 - Remove Extra Tasks
# =====================================================

# Keep only the first toxicity prediction task

train_y = train_y[:, 0]
valid_y = valid_y[:, 0]
test_y = test_y[:, 0]

train_w = train_w[:, 0]
valid_w = valid_w[:, 0]
test_w = test_w[:, 0]

print("\nTask reduction complete.")

print("Training labels:", train_y.shape)
print("Validation labels:", valid_y.shape)
print("Testing labels:", test_y.shape)

print("\nDataset Summary")
print("-" * 40)

print(f"Training Features   : {train_X.shape}")
print(f"Training Labels     : {train_y.shape}")

print(f"Validation Features : {valid_X.shape}")
print(f"Validation Labels   : {valid_y.shape}")

print(f"Testing Features    : {test_X.shape}")
print(f"Testing Labels      : {test_y.shape}")

# =====================================================
# Step 4 - Define TensorFlow Graph
# =====================================================

tf.compat.v1.disable_eager_execution()

d = 1024
n_hidden = 50
learning_rate = 0.001
n_epochs = 10
batch_size = 100

with tf.name_scope("placeholders"):

    x = tf.compat.v1.placeholder(
        tf.float32,
        shape=(None, d),
        name="x"
    )

    y = tf.compat.v1.placeholder(
        tf.float32,
        shape=(None,),
        name="y"
    )

print("\nTensorFlow graph initialized.")

print(f"Hidden Neurons : {n_hidden}")
print(f"Learning Rate  : {learning_rate}")
print(f"Epochs         : {n_epochs}")
print(f"Batch Size     : {batch_size}")

# =====================================================
# Step 5 - Implement Hidden Layer
# =====================================================

with tf.name_scope("hidden-layer"):

    W = tf.Variable(
        tf.random.normal((d, n_hidden)),
        name="weights"
    )

    b = tf.Variable(
        tf.random.normal((n_hidden,)),
        name="bias"
    )

    x_hidden = tf.nn.relu(
        tf.matmul(x, W) + b
    )

print("\nHidden layer created successfully.")
print(f"Weight matrix shape : {W.shape}")
print(f"Bias vector shape   : {b.shape}")

# =====================================================
# Step 6 - Add Dropout to Hidden Layer
# =====================================================

with tf.name_scope("dropout"):

    x_hidden = tf.nn.dropout(
        x_hidden,
        rate=0.5,
        name="dropout"
    )

print("\nDropout layer added successfully.")
print("Dropout Rate : 50%")

# =====================================================
# Step 7 - Complete the Fully Connected Architecture
# =====================================================

with tf.name_scope("output"):

    W_out = tf.Variable(
        tf.random.normal((n_hidden, 1)),
        name="output_weights"
    )

    b_out = tf.Variable(
        tf.random.normal((1,)),
        name="output_bias"
    )

    y_logit = tf.matmul(x_hidden, W_out) + b_out

    # Probability of toxicity

    y_one_prob = tf.sigmoid(y_logit)

    # Binary prediction

    y_pred = tf.round(y_one_prob)

print("\nOutput layer created successfully.")
print(f"Output weight matrix : {W_out.shape}")
print(f"Output bias vector   : {b_out.shape}")

# =====================================================
# Step 8 - Define the Loss Function
# =====================================================

with tf.name_scope("loss"):

    y_expand = tf.expand_dims(y, 1)

    entropy = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=y_logit,
        labels=y_expand
    )

    l = tf.reduce_sum(entropy)

print("\nLoss function created successfully.")

# =====================================================
# Step 9 - Configure the Optimizer
# =====================================================

with tf.name_scope("optim"):

    train_op = tf.compat.v1.train.AdamOptimizer(
        learning_rate
    ).minimize(l)

print("Adam optimizer initialized.")

# =====================================================
# Step 10 - Configure TensorBoard Summaries
# =====================================================

with tf.name_scope("summaries"):

    tf.compat.v1.summary.scalar("loss", l)
    merged = tf.compat.v1.summary.merge_all()

print("TensorBoard summaries created.")

# =====================================================
# Step 11 - Implement Mini-Batch Training
# =====================================================

# Create TensorBoard writer
train_writer = tf.compat.v1.summary.FileWriter(
    "tensorboard/tox21",
    tf.compat.v1.get_default_graph()
)

# Number of training samples
N = train_X.shape[0]

# Start TensorFlow session
with tf.compat.v1.Session() as sess:

    # Initialize weights and biases
    sess.run(tf.compat.v1.global_variables_initializer())

    step = 0

    # Training loop
    for epoch in range(n_epochs):

        print(f"\n========== Epoch {epoch + 1}/{n_epochs} ==========")

        pos = 0
        batch = 0

        while pos < N:

            # Select mini-batch
            batch_X = train_X[pos:pos + batch_size]
            batch_y = train_y[pos:pos + batch_size]

            feed_dict = {
                x: batch_X,
                y: batch_y
            }

            # Train one batch
            _, summary, loss = sess.run(
                [train_op, merged, l],
                feed_dict=feed_dict
            )

            # Save TensorBoard summary
            train_writer.add_summary(summary, step)

            # Print progress every 10 batches
            if batch % 10 == 0:
                print(
                    f"Batch {batch:2d} | "
                    f"Samples {pos:4d}-{min(pos + batch_size, N):4d} | "
                    f"Loss = {loss:.4f}"
                )

            step += 1
            batch += 1
            pos += batch_size

        print(f"Epoch {epoch + 1} complete.")
    # =====================================================
    # Step 12 - Generate Predictions
    # =====================================================

    print("\nGenerating predictions on validation dataset...")

    valid_y_pred = sess.run(
        y_pred,
        feed_dict={
            x: valid_X
        }
    )

    print("Predictions generated successfully.")
    print(f"Prediction shape: {valid_y_pred.shape}")

    # =====================================================
    # Step 13 - Evaluate Model Accuracy
    # =====================================================

    valid_y_pred = valid_y_pred.flatten()

    accuracy = accuracy_score(
        valid_y,
        valid_y_pred
    )

    print("\nValidation Accuracy")
    print("-" * 30)
    print(f"Accuracy: {accuracy:.4f}")

    # Close the TensorBoard writer
    train_writer.close()

print("\nTraining complete.")