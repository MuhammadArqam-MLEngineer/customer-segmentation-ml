"""
preprocessing.py

Handles data loading and preprocessing for the customer behavior dataset.

Functions:
- load_and_preprocess_data(file_path): 
    Loads the dataset, cleans column names, defines numerical and categorical features, 
    and builds a preprocessing pipeline that imputes missing values, scales numerical data, 
    and one-hot encodes categorical variables.

Returns:
- df: Raw Pandas DataFrame.
- X: Features before transformation.
- processor: A ColumnTransformer pipeline ready for modeling.

Author: Muhammad Arqam
"""



import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_and_preprocess_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]

    # Define columns
    num_data = ["age", "total_earning", "total_spend", "items_purchased", "average_rating", "days_since_last_purchase"]
    cat_data = ["gender", "membership_type", "discount_applied", "satisfaction_level"]

    # Pipelines
    num_transforming = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    cat_transforming = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    processor = ColumnTransformer(transformers=[
        ("num", num_transforming, num_data),
        ("cat", cat_transforming, cat_data)
    ])

    X = df[num_data + cat_data]

    return df, X, processor
