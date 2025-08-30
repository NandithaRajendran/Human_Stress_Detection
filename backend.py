import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load trained models
with open('rf_boost_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('svm_boost_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

with open('voting_clf_model.pkl', 'rb') as f:
    voting_model = pickle.load(f)

# ✅ Load the exact scaler used during training
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def hybrid_predict(input_data: pd.DataFrame):
    # Convert input to numpy and scale using trained scaler
    input_array = scaler.transform(input_data)

    # Make predictions
    rf_pred = rf_model.predict(input_array)[0]
    svm_pred = svm_model.predict(input_array)[0]
    voting_pred = voting_model.predict(input_array)[0]

    predictions = np.array([rf_pred, svm_pred, voting_pred])
    majority_vote = np.bincount(predictions).argmax()

    return majority_vote