#This program loads pre trained model trained in main.ipynb and stored in models directory and uses it to predict whether the newly simulated latest transaction (later saved in records collection in mongodb database) is a fraud or a legit transaction
#This program aims to simulate a credit card transaction
#THis program also uses SHAP to explain the model predictions and give confidence score
import streamlit as st
import pandas as pd
import random
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pymongo import MongoClient
import xgboost as xgb


model = joblib.load("models/xgb_fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")
target_encoder = joblib.load("models/target_encoder.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")


client = MongoClient("mongodb://localhost:27017/")
db = client["creditcard_db"]
collection = db["records"]


card_types = ["Visa", "MasterCard", "Discover"]
locations = ["NY", "CA", "TX", "WA"]
txn_types = ["Online", "InStore"]
merchant_ids = [f"M{random.randint(1000, 9999)}" for _ in range(100)]


def generate_transaction():
    return {
        "Time": random.randint(1000, 100000),
        "Amount": round(random.uniform(1.0, 1000.0), 2),
        "MerchantID": random.choice(merchant_ids),
        "Location": random.choice(locations),
        "CardType": random.choice(card_types),
        "TransactionType": random.choice(txn_types),
        "IsForeignTransaction": random.choice([0, 1]),
        "IsHighRiskCountry": random.choice([0, 1]),
    }


def preprocess_for_model(txn_dict):
    df = pd.DataFrame([txn_dict])

    
    df["MerchantID"] = target_encoder.transform(df["MerchantID"])

    
    df = pd.get_dummies(df, columns=["Location", "CardType", "TransactionType"], drop_first=True)

    
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    
    df = df[feature_columns]

    
    df_scaled = scaler.transform(df)
    return df_scaled, df


st.title("Credit Card Fraud Detection Module")


if st.button("Generate Transaction"):
    new_txn = generate_transaction()
    st.json(new_txn)

    
    collection.insert_one(new_txn)
    st.success("Transaction inserted into MongoDB")


latest_txn = collection.find().sort("_id", -1).limit(1)
latest_txn = list(latest_txn)
if latest_txn:
    st.subheader("Latest Transaction")
    txn = latest_txn[0]
    txn.pop("_id", None)
    st.json(txn)

    if st.button("Predict Fraud"):
        X_scaled, X_df = preprocess_for_model(txn)
        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0][1]

        st.markdown(f"Prediction: {'FRAUD' if pred == 1 else 'LEGIT'}")
        st.markdown(f"Confidence (Fraud Probability)**: {proba:.4f}")

        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_df)

        st.subheader("SHAP Force Plot")
        shap.initjs()
       
        shap_html = shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        X_df.iloc[0],
        feature_names=X_df.columns
        )

        st.components.v1.html(shap.getjs() + shap_html.html(), height=300)


        
        shap_df = pd.DataFrame({
            'feature': X_df.columns,
            'shap_value': shap_values[0]
        }).sort_values(by="shap_value", key=abs, ascending=False)

        st.subheader("Top Contributing Features")
        st.write(shap_df.head(5))
