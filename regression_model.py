"""
regression_model.py

Trains and evaluates a regression model to predict customer spending 
based on behavioral features and cluster assignments.

Functions:
- perform_regression(df):
    Splits the dataset into train/test sets, fits a Random Forest Regressor, 
    evaluates model performance using R² and MSE, and visualizes predictions.

Returns:
- model: The trained regression model instance.

Author: Muhammad Arqam
"""



import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def perform_regression(df):
    X = df[["items_purchased", "average_rating", "Clusters"]]
    y = df["total_spend"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate model
    print("R2 Score:", r2_score(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))

    # Visualization
    plt.scatter(y_test, y_pred)
    plt.title("Actual vs Predicted")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.show()

    return model
