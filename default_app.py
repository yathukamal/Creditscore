#Set up app for streamlit to deploy for external users
#import libraries
import streamlit as st
import joblib
import pandas as pd

#-------------------------------
#Load trained model and scaler
#-------------------------------
model = joblib.load("default_log_reg_model.joblib")
scaler = joblib.load("default_scaler.joblib")

#-------------------------------
#App title
#-------------------------------
st.title("Credit Default Predictor")
st.write("Adjust the inputs below to predict the likelihood of a customer defaulting ")

#-------------------------------
#User input fields
#-------------------------------
income = st.number_input(
    "Total income (£)",
    min_value=0,
    max_value=120_000,
    value=30_000,
    step=1_000
)

savings = st.number_input(
    "Total savings (£)",
    min_value=0,
    max_value=200_000,
    value=10_000,
    step=1_000
)

debt = st.number_input(
    "Total debt (£)",
    min_value=0,
    max_value=100_000,
    value=2_000,
    step=1_000
)

#income = st.number_input("Total income (£)", min_value=0, value=30000)
#savings = st.number_input("Total savings (£)", min_value=0, value=50000)
#debt = st.number_input("Total debt (£)", min_value=0, value=2000)

cat_gambling = st.selectbox("Do you gamble and if so, how frequently?", ["no","low","high"])
cat_credit_card = st.selectbox("Do you have a credit card?", ["yes","no"])

#Compute rations using user inputs
r_savings_income = savings / income if income > 0 else 0
r_debt_income = debt / income if income > 0 else 0
r_debt_savings = debt / savings if savings > 0 else 0


#-------------------------------
#Create input DataFrame
#-------------------------------
input_df = pd.DataFrame([{
    "INCOME": income,
    "SAVINGS": savings,
    "DEBT": debt,
    "R_SAVINGS_INCOME": r_savings_income,
    "R_DEBT_INCOME": r_debt_income,
    "R_DEBT_SAVINGS": r_debt_savings,
    "CAT_DEBT": int(debt > 0),
    "CAT_GAMBLING": cat_gambling,
    "CAT_CREDIT_CARD": int(cat_credit_card == "yes")
}])

#-------------------------------
#One-hot encode categorical variables
#-------------------------------
input_df = pd.get_dummies(input_df, columns=["CAT_GAMBLING"], prefix="GAMBLING")

for col in scaler.feature_names_in_:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[scaler.feature_names_in_]

#-------------------------------
#Scale and predict
#-------------------------------
#Scale
scaled_input = scaler.transform(input_df)

#Predict
prediction = model.predict(scaled_input)[0]
pred_prob = model.predict_proba(scaled_input)[0, 1]

#-------------------------------
#Display results
#-------------------------------
st.subheader("Prediction")

if st.button("Predict default risk"):
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    pred_prob = model.predict_proba(scaled_input)[0, 1]

    if prediction == 1:
        st.error("⚠️High risk of default")
    else:
        st.success("✅Low risk of default")

    st.write(f"Probability of default:{pred_prob:.2f}")