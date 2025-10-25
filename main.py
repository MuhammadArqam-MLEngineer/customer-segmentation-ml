"""
main.py

This script serves as the entry point for the Customer Segmentation and Spending Prediction project.

Workflow:
1. Loads and preprocesses raw customer data.
2. Performs K-Means clustering to group customers by behavioral patterns.
3. Trains a regression model to predict customer spending based on features and cluster assignments.

Author: Muhammad Arqam
"""


from preprocessing import load_and_preprocess_data
from clustering_model import perform_clustering
from regression_model import perform_regression

def main():
    file_path = "ecommerce_customer_behaviour.csv"
    
    # Step 1: Preprocessing
    df, X, processor = load_and_preprocess_data(file_path)
    
    # Step 2: Clustering
    df = perform_clustering(df, X, processor)
    
    # Step 3: Regression
    perform_regression(df)

if __name__ == "__main__":
    main()
