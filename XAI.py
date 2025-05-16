#This program uses explainable AI namely SHAP and LIME to explain the model predictions the dataset is retrieved from mongodb database
#previously generated in the new_dataset_generate.py program and uses the xgboost model to run the prediction and saves the plots in the plots folder
import os
import pandas as pd
import joblib
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from pymongo import MongoClient


os.makedirs("plots", exist_ok=True)


model = joblib.load("models/xgb_fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")
target_encoder = joblib.load("models/target_encoder.pkl")


feature_names = model.get_booster().feature_names


client = MongoClient("mongodb://localhost:27017/")
db = client["creditcard_db"]
collection = db["test_transactions"]
df = pd.DataFrame(list(collection.find()))
print(" Data loaded from MongoDB")


df.drop(columns=['_id'], inplace=True)
y = df['IsFraud']
X = df.drop(columns=['IsFraud'])


X[['MerchantID']] = target_encoder.transform(X[['MerchantID']])


X = pd.get_dummies(X, columns=['Location', 'CardType', 'TransactionType'], drop_first=True)


X = X.reindex(columns=feature_names, fill_value=0)


X_scaled = scaler.transform(X)


print(" Running SHAP...")


explainer_shap = shap.TreeExplainer(model)
shap_values = explainer_shap.shap_values(X_scaled)


shap.summary_plot(shap_values, X, show=False)
plt.savefig("plots/shap_summary_2.png")
plt.close()


shap.force_plot(
    explainer_shap.expected_value, 
    shap_values[0], 
    X.iloc[0], 
    matplotlib=True, 
    show=False
)
plt.savefig("plots/shap_force_instance_02.png")
plt.close()

print(" SHAP visualizations saved.")


print(" Running LIME...")


lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_scaled,
    feature_names=feature_names,
    class_names=["Not Fraud", "Fraud"],
    verbose=True,
    mode="classification",
    kernel_width=3.0  
)


lime_exp = lime_explainer.explain_instance(X_scaled[0], model.predict_proba, num_features=10)


lime_exp.save_to_file("plots/lime_instance_0.html")
print(" LIME explanation saved to HTML (open plots/lime_instance_0.html in browser).")
