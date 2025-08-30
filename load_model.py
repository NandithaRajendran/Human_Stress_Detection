import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# Load the dataset
data = pd.read_csv("SaYoPillow.csv")

# Rename columns for consistency (make lowercase and strip spaces)
data.columns = [col.strip().lower().replace(' ', '_') for col in data.columns]

# Check if the expected columns exist
expected_cols = ['sr', 'rr', 't', 'sl', 'hr']
missing = [col for col in expected_cols if col not in data.columns]

if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

# Select input features
X = data[expected_cols]

# Fit the scaler
scaler = StandardScaler()
scaler.fit(X)

# Save the scaler to a file
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ scaler.pkl created successfully.")