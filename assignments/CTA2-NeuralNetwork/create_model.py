import pandas as pd

from keras import Input
from keras.models import Sequential
from keras.layers import Dense

# --------------------------------------------------
# Load the scaled training data
# --------------------------------------------------
training_data_df = pd.read_csv("sales_data_training_scaled.csv")

# --------------------------------------------------
# Split the data into:
# X = input features (9 columns)
# Y = target value (total_earnings)
# --------------------------------------------------
X = training_data_df.drop("total_earnings", axis=1).values
Y = training_data_df[["total_earnings"]].values

# Verify the data loaded correctly
print("X shape:", X.shape)
print("Y shape:", Y.shape)

# --------------------------------------------------
# Build the Neural Network
# --------------------------------------------------

# Create an empty sequential model
model = Sequential(name="VideoGameSalesPredictor")

# Define the input layer
# Each game has 9 input features
model.add(Input(shape=(9,)))

# Hidden Layer 1
# Learns basic relationships in the data
model.add(
    Dense(
        units=50,
        activation="relu"
    )
)

# Hidden Layer 2
# Learns more complex combinations of features
model.add(
    Dense(
        units=100,
        activation="relu"
    )
)

# Hidden Layer 3
# Refines the learned patterns
model.add(
    Dense(
        units=50,
        activation="relu"
    )
)

# Output Layer
# One output because we're predicting one continuous value:
# total_earnings
model.add(
    Dense(
        units=1,
        activation="linear"
    )
)

# Display the network architecture
model.summary()

# --------------------------------------------------
# Configure how the neural network will learn
# --------------------------------------------------
model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

model.fit(
    X,
    Y,
    epochs=50,
    shuffle=True,
    verbose=2
)

# --------------------------------------------------
# Load the scaled test data
# --------------------------------------------------
test_data_df = pd.read_csv("sales_data_test_scaled.csv")

X_test = test_data_df.drop("total_earnings", axis=1).values
Y_test = test_data_df[["total_earnings"]].values

# --------------------------------------------------
# Evaluate the neural network
# --------------------------------------------------
test_error_rate = model.evaluate(
    X_test,
    Y_test,
    verbose=0
)

print("\nTest Mean Squared Error (MSE):", test_error_rate)

# --------------------------------------------------
# Save the trained model
# --------------------------------------------------
model.save("trained_model.h5")

print("Model saved to disk.")