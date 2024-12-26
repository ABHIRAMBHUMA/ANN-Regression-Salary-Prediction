import streamlit as st  # type: ignore
import numpy as np
import pandas as pd
import tensorflow as tf  # type: ignore
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder  # type: ignore
import pickle

# Load the trained model
model = tf.keras.models.load_model('regression_model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app title
st.title('Estimated Salary Prediction')

# User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
exited = st.selectbox('Exited', [0, 1], key="exited_unique_key")
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1], key="credit_card_key")
is_active_number = st.selectbox('Is Active Member', [0, 1], key="active_member_key")

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_number],
    'Exited': [exited]
})

# Transform the geography input using OneHotEncoder
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

# Convert the one-hot encoded data to a DataFrame
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine it with the input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict the estimated salary
predicted_salary = model.predict(input_data_scaled)
predicted_salary_value = predicted_salary[0][0]

# Display the predicted salary
st.write(f'Predicted Estimated Salary: ${predicted_salary_value:,.2f}')
