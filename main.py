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
from sklearn.model_selection import train_test_split

def main():
    file_path = "ecommerce_customer_behaviour.csv"
    
    # Step 1: Preprocessing
    df, X, processor , num_data , cat_data = load_and_preprocess_data(file_path)
    
    # Step 2: Split data before clustering to avoid data leakage
    train_idx , test_idx = train_test_split(range(len(df)) , test_size=0.2 , random_state=42)
    
    # Create train and test sets using the indices
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    
    # Step 3: Clustering (fit only on training data)
    clf, train_clusters, test_clusters = perform_clustering(X_train, X_test, processor)
    
    # Add clusters to dataframe for visualization
    df_train = df.iloc[train_idx].copy()
    df_train["Clusters"] = train_clusters
    
    # Optional: Visualize clusters
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_train, x="items_purchased", y="total_spend", 
                   hue="Clusters", palette="viridis", alpha=0.6)
    plt.title("Customer Clusters")
    plt.show()
    
    # Step 4: Regression
    model, feature_names = perform_regression(df, train_idx, test_idx, 
                                             train_clusters, test_clusters)
    
    print("\n Model traning Complete!")

if __name__ == "__main__":
    main()
