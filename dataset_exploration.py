#This program connects to a MongoDB database, retrieves a dataset of credit card transactions, and performs exploratory data analysis (EDA) on the dataset.
# It generates various plots to visualize the data, including distributions of transaction amounts and times, class distributions, and correlations between features.
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient


MONGO_URI = "mongodb://localhost:27017/"  
DB_NAME = "creditcard_db"  
COLLECTION_NAME = "transactions"  


output_dir = "plots"  
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


try:
    
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    
    cursor = collection.find()
    
    
    df = pd.DataFrame(list(cursor))
    
    print("Data successfully loaded from MongoDB!")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    exit()


df.drop(columns=['_id'], inplace=True)


print("First 5 rows of the dataset:")
print(df.head())


print("\nGeneral Information about the dataset:")
df_info = df.info()


print("\nDescriptive statistics of numerical columns:")
print(df.describe())


print("\nMissing values in each column:")
missing_values = df.isnull().sum()
print(missing_values)


print("\nClass distribution of 'IsFraud' (fraud vs non-fraud transactions):")
fraud_distribution = df['IsFraud'].value_counts(normalize=True)
print(fraud_distribution)


plt.figure(figsize=(10, 6))
sns.histplot(df['Amount'], bins=50, kde=True)
plt.title("Distribution of Transaction Amounts")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.savefig(os.path.join(output_dir, "transaction_amount_distribution.png"))
plt.close()  


plt.figure(figsize=(10, 6))
sns.histplot(df['Time'], bins=100, kde=True)
plt.title("Distribution of Transaction Times")
plt.xlabel("Time (seconds in a day)")
plt.ylabel("Frequency")
plt.savefig(os.path.join(output_dir, "transaction_time_distribution.png"))
plt.close()


plt.figure(figsize=(10, 6))
sns.boxplot(x='IsFraud', y='Amount', data=df)
plt.title("Amount Distribution by Fraud Class (Fraud vs Non-Fraud)")
plt.xlabel("Fraud Class (0 = Non-Fraud, 1 = Fraud)")
plt.ylabel("Amount")
plt.savefig(os.path.join(output_dir, "amount_by_fraud_class.png"))
plt.close()


numerical_df = df.select_dtypes(include=[np.number])

plt.figure(figsize=(10, 6))
correlation_matrix = numerical_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Numerical Features")
plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
plt.close()


categorical_columns = ['MerchantID', 'Location', 'CardType', 'TransactionType']
for col in categorical_columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=col, data=df, hue='IsFraud')
    plt.title(f"Countplot of {col} by Fraud Class")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.savefig(os.path.join(output_dir, f"{col}_by_fraud_class.png"))
    plt.close()


plt.figure(figsize=(10, 6))
merchant_counts = df['MerchantID'].value_counts().head(20)  
sns.barplot(x=merchant_counts.index, y=merchant_counts.values)
plt.title("Top 20 Merchants by Transaction Count")
plt.xlabel("Merchant ID")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.savefig(os.path.join(output_dir, "top_20_merchants.png"))
plt.close()


plt.figure(figsize=(10, 6))
sns.kdeplot(df[df['IsFraud'] == 0]['Time'], label='Non-Fraud', shade=True)
sns.kdeplot(df[df['IsFraud'] == 1]['Time'], label='Fraud', shade=True)
plt.title("Density Plot of Transaction Times by Fraud Class")
plt.xlabel("Time (seconds in a day)")
plt.ylabel("Density")
plt.legend()
plt.savefig(os.path.join(output_dir, "transaction_time_by_fraud_class.png"))
plt.close()
