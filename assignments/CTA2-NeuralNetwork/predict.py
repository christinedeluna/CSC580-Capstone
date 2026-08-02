import pandas as pd
from keras.models import load_model

# --------------------------------------------------
# Load the trained model
# --------------------------------------------------
model = load_model("trained_model.h5")

print("Model loaded successfully.")

# --------------------------------------------------
# Load the new product
# --------------------------------------------------
X = pd.read_csv("proposed_new_product.csv").values

print("\nNew product features:")
print(X)

# --------------------------------------------------
# Predict earnings
# --------------------------------------------------
prediction = model.predict(X)

# Grab the first prediction
prediction = prediction[0][0]

print("\nScaled prediction:", prediction)

# --------------------------------------------------
# Convert back to dollars
# --------------------------------------------------
prediction = prediction + 0.115913
prediction = prediction / 0.0000036968

print("\nPredicted Total Earnings: ${:,.2f}".format(prediction))