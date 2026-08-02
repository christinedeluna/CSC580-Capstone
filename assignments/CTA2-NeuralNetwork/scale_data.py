import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# -------------------------
# Load the original datasets
# -------------------------
training_data_df = pd.read_csv("sales_data_training.csv")
test_data_df = pd.read_csv("sales_data_test.csv")

# -------------------------
# Scale all columns between 0 and 1
# -------------------------
scaler = MinMaxScaler(feature_range=(0, 1))

scaled_training = scaler.fit_transform(training_data_df)
scaled_testing = scaler.transform(test_data_df)

# -------------------------
# Display the scaling values
# for total_earnings
# -------------------------
print(
    "Note: total_earnings values were scaled by "
    "multiplying by {:.10f} and adding {:.6f}".format(
        scaler.scale_[8],
        scaler.min_[8]
    )
)

# -------------------------
# Convert back to DataFrames
# -------------------------
scaled_training_df = pd.DataFrame(
    scaled_training,
    columns=training_data_df.columns
)

scaled_testing_df = pd.DataFrame(
    scaled_testing,
    columns=test_data_df.columns
)

# -------------------------
# Save scaled datasets
# -------------------------
scaled_training_df.to_csv(
    "sales_data_training_scaled.csv",
    index=False
)

scaled_testing_df.to_csv(
    "sales_data_test_scaled.csv",
    index=False
)

print("\nScaled data files created successfully!")