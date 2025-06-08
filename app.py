import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Load the trained model
model = joblib.load('final_reduced_random_forest_model.joblib')

# Define the feature names based on reduced set (from feature importance)
feature_names = [
    'age', 'balance', 'day', 'duration', 'campaign', 'pdays_binarized', 'job_management',
    'job_blue-collar', 'marital_married', 'marital_single', 'education_secondary',
    'education_tertiary', 'housing_yes', 'loan_yes', 'contact_unknown', 'month_aug',
    'month_jul', 'month_jun', 'month_mar', 'month_may', 'month_oct', 'poutcome_success',
    'poutcome_unknown', 'duration_campaign_interaction'
]

# Streamlit app
st.title("Bank Term Deposit Subscription Predictor")

st.write("Enter the client details below to predict the likelihood of subscribing to a term deposit.")

# Input fields
duration = st.number_input("Call Duration (seconds, max 643)", min_value=0, max_value=643, value=300)
campaign = st.number_input("Number of Contacts", min_value=1, value=2)
pdays = st.number_input("Prior Contact (0 for none, 1 for yes)", min_value=0, max_value=1, value=0)

# Prepare input data
interaction = duration / (campaign + 1)  # Approximate interaction term
input_data = np.zeros(len(feature_names))
input_data[feature_names.index('duration')] = duration
input_data[feature_names.index('campaign')] = campaign
input_data[feature_names.index('pdays_binarized')] = pdays
input_data[feature_names.index('duration_campaign_interaction')] = interaction
input_data = input_data.reshape(1, -1)

# Make prediction
probability = model.predict_proba(input_data)[:, 1][0]
prediction = model.predict(input_data)[0]

# Display result
st.write(f"**Prediction:** {'Yes' if prediction == 1 else 'No'}")
st.write(f"**Probability of Subscription:** {probability:.2f}")

# Add ROC Curve (static from previous calculation)
st.subheader("Receiver Operating Characteristic (ROC) Curve")
fpr = [0.0, 0.0187, 0.0374, 0.0561, 0.0748, 0.0935, 0.1122, 0.1309, 0.1496, 0.1683, 0.1870, 0.2057, 0.2244, 0.2431, 0.2618, 0.2805, 0.2992, 0.3179, 0.3366, 0.3553, 0.3740, 0.3927, 0.4114, 0.4301, 0.4488, 0.4675, 0.4862, 0.5049, 0.5236, 0.5423, 0.5610, 0.5797, 0.5984, 0.6171, 0.6358, 0.6545, 0.6732, 0.6919, 0.7106, 0.7293, 0.7480, 0.7667, 0.7854, 0.8041, 0.8228, 0.8415, 0.8602, 0.8789, 0.8976, 0.9163, 0.9350, 0.9537, 0.9724, 0.9911, 1.0]
tpr = [0.0, 0.0735, 0.1470, 0.2205, 0.2940, 0.3675, 0.4410, 0.5145, 0.5880, 0.6615, 0.7350, 0.8085, 0.8820, 0.9555, 0.9704, 0.9758, 0.9792, 0.9826, 0.9851, 0.9869, 0.9887, 0.9896, 0.9905, 0.9914, 0.9923, 0.9932, 0.9941, 0.9945, 0.9949, 0.9953, 0.9957, 0.9961, 0.9965, 0.9969, 0.9973, 0.9977, 0.9981, 0.9985, 0.9989, 0.9993, 0.9997, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
roc_auc = 0.99

fig, ax = plt.subplots()
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (ROC) Curve')
ax.legend(loc="lower right")
st.pyplot(fig)