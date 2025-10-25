"""
clustering_model.py

Implements customer segmentation using K-Means clustering.

Functions:
- perform_clustering(df, X, processor, n_clusters=3):
    Applies the preprocessing pipeline, fits a K-Means model, and assigns cluster labels
    back to the original dataset. Also visualizes clusters by membership type and rating.

Returns:
- df: DataFrame with an added 'Clusters' column for each record.

Author: Muhammad Arqam
"""



import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score

def perform_clustering(df, X, processor, n_clusters=3):
    clf = Pipeline(steps=[
        ("processor", processor),
        ("model", KMeans(n_clusters=n_clusters, random_state=42))
    ])

    clf.fit(X)
    df["Clusters"] = clf["model"].labels_

    score = silhouette_score(clf["processor"].transform(X), clf["model"].labels_)
    print(f"Silhouette Score: {score:.3f}")

    # Visualization
    sns.scatterplot(x="membership_type", y="average_rating", hue="Clusters", data=df)
    plt.title("Customer Clusters by Membership Type and Rating")
    plt.show()

    return df
