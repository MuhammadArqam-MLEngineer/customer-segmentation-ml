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
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.metrics import mean_squared_error, r2_score , mean_absolute_error

def perform_regression(df , train_idx , test_idx , train_clusters , test_clusters):

    # Use more features for regression
    feature_cols = ["age", "total_earning", "items_purchased", 
                   "average_rating", "days_since_last_purchase",
                   "gender", "membership_type", "discount_applied", 
                   "satisfaction_level"]
    
    X = df[feature_cols].copy()
    y = df["total_spend"]

    #split data
    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    #Add cluster labels as features
    X_train["Clusters"] = train_clusters
    X_test["Clusters"] = test_clusters

    #Encode catergorical variables
    X_train_encoded = pd.get_dummies(X_train , columns=["gender", "membership_type", 
                                                        "discount_applied", "satisfaction_level"])
    X_test_encoded = pd.get_dummies(X_test , columns=["gender", "membership_type", 
                                                      "discount_applied", "satisfaction_level"])
    
    #Align columns
    X_train_encoded , X_test_encoded = X_train_encoded.align(X_test_encoded , join="left" , axis=1 , fill_value=0)

    #Train model with better parameters
    model = RandomForestRegressor(n_estimators=200 , max_depth=10 , min_samples_split=5 , random_state=42 , n_jobs=-1)
    
    model.fit(X_train_encoded , y_train)

    # cross validation on training set
    cv_scores = cross_val_score(model , X_train_encoded , y_train , cv=5 , scoring='r2' , n_jobs=-1)
    
    print(f"Cross Validation R2 Score : {cv_scores}")
    print(f"Mean CV R2 : {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    #Predictions
    y_pred = model.predict(X_test_encoded)

    #Evaluate Model 
    print("\n Test Set Performance:")
    print(f"R2 Score: {r2_score(y_test , y_pred):.4f}")
    print(f"RMSE : {np.sqrt(mean_squared_error(y_test , y_pred)):.2f}")
    print(f"MAE : {mean_absolute_error(y_test , y_pred):.2f}")

    #Feature Importance
    feature_importance = pd.DataFrame({
        'feature' : X_train_encoded.columns,
        'importance' : model.feature_importances_
    }).sort_values('importance' , ascending=False)

    print("\nTop 10 Important Features")
    print(feature_importance.head(10))

    # Visualization
    plt.figure(figsize=(10 , 5))
    plt.subplot(1 , 2 , 1)
    plt.scatter(y_test ,y_pred , alpha=0.5)
    plt.plot([y_test.min() , y_test.max()] , [y_test.min() , y_test.max()] , 'r--' , lw=2)
    plt.title("Actual vs Predicted")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")

    plt.subplot(1 , 2 ,1)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title("Residual Plot")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")

    plt.tight_layout()
    plt.show()

    return model , X_test_encoded.columns