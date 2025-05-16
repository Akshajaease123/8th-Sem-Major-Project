#This program is used to synthetically generate a new dataset by taking in a reference dataset from csv file and automatically
#saves the dataset in mongodb database
import pandas as pd
import numpy as np
import random
from pymongo import MongoClient


def generate_transaction_from_existing(row, is_fraud):
    if is_fraud:
        time = np.random.randint(0, 21600)  
        amount = round(np.random.normal(200, 100), 2)
        transaction_type = np.random.choice(["Online", "InStore"], p=[0.85, 0.15])
        is_foreign = np.random.choice([0, 1], p=[0.3, 0.7])
        is_high_risk_country = np.random.choice([0, 1], p=[0.6, 0.4])
    else:
        time = np.random.randint(0, 86400)
        amount = round(np.random.normal(50, 30), 2)
        transaction_type = np.random.choice(["Online", "InStore"], p=[0.3, 0.7])
        is_foreign = np.random.choice([0, 1], p=[0.95, 0.05])
        is_high_risk_country = np.random.choice([0, 1], p=[0.98, 0.02])

    return {
        "Time": max(0, min(86399, time)),
        "Amount": max(0.01, amount),
        "MerchantID": f"M{np.random.randint(1000, 9999)}",  
        "Location": random.choice(["NY", "CA", "TX", "FL", "WA", "NV", "IL", "AZ"]),
        "CardType": random.choice(["Visa", "MasterCard", "Amex", "Discover"]),
        "TransactionType": transaction_type,
        "IsForeignTransaction": is_foreign,
        "IsHighRiskCountry": is_high_risk_country,
        "IsFraud": int(is_fraud)
    }


csv_file = "fraudTest.csv" 
df = pd.read_csv(csv_file)


df = df.drop(columns=['Unnamed: 0', 'trans_date_trans_time'])


total_records = len(df)
fraud_ratio = 0.05
num_frauds = int(total_records * fraud_ratio)
num_nonfrauds = total_records - num_frauds


fraud_data = [generate_transaction_from_existing(df.iloc[i], True) for i in range(num_frauds)]
nonfraud_data = [generate_transaction_from_existing(df.iloc[i], False) for i in range(num_nonfrauds)]
all_data = fraud_data + nonfraud_data
random.shuffle(all_data)


generated_df = pd.DataFrame(all_data)


output_csv = "generated_creditcard_test_data.csv"
generated_df.to_csv(output_csv, index=False)
print(f" Generated data saved as: {output_csv}")


MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "creditcard_db"
COLLECTION_NAME = "test_transactions"

try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]


    collection.delete_many({})

    
    records = generated_df.to_dict(orient="records")
    collection.insert_many(records)
    print(f" Uploaded {len(records)} transactions to MongoDB collection '{COLLECTION_NAME}' in database '{DB_NAME}'.")
except Exception as e:
    print(f" MongoDB upload failed: {e}")
