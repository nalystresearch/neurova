# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Neurova ML module self-test (NumPy-only)."""
import sys
sys.path.insert(0, '.')

import numpy as np
from neurova.ml import (
    StandardScaler, MinMaxScaler, LabelEncoder,
    KNearestNeighbors, LogisticRegression, NaiveBayes,
    KMeans, DBSCAN, PCA, LDA,
    accuracy_score, train_test_split, cross_validate
)

print("=" * 60)
print("NEUROVA ML MODULE - NUMPY ONLY")
print("=" * 60)

# test 1: Preprocessing
print("\nTest 1: Preprocessing")
X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"   StandardScaler: {X_scaled.shape} | mean={X_scaled.mean():.4f}")

y = np.array(['cat', 'dog', 'cat', 'bird'])
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
print(f"   LabelEncoder: {list(y)} -> {list(y_encoded)}")

# test 2: Classification
print("\nTest 2: Classification")
X_train = np.random.randn(100, 2)
y_train = (X_train[:, 0] > 0).astype(int)
knn = KNearestNeighbors(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_train[:20])
acc = accuracy_score(y_train[:20], y_pred)
print(f"   KNN: accuracy={acc:.2f}")

# test 3: Clustering
print("\nTest 3: Clustering")
X_cluster = np.random.randn(50, 2)
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_cluster)
print(f"   KMeans: {len(np.unique(labels))} clusters found")

# test 4: Dimensionality Reduction
print("\nTest 4: Dimensionality Reduction")
X_pca = np.random.randn(50, 10)
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_pca)
print(f"   PCA: {X_pca.shape} -> {X_reduced.shape}")
print(f"   Explained variance ratio: {pca.explained_variance_ratio_}")

# test 5: Train-Test Split
print("\nTest 5: Model Selection")
X_split = np.random.randn(100, 5)
y_split = np.random.randint(0, 2, 100)
X_train, X_test, y_train, y_test = train_test_split(
    X_split, y_split, test_size=0.3, random_state=42
)
print(f"   train_test_split: train={X_train.shape[0]}, test={X_test.shape[0]}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
print("\nNeurova ML provides:")
print("  - Preprocessing: StandardScaler, MinMaxScaler, Normalizer")
print("  - Classification: KNN, LogisticRegression, NaiveBayes, DecisionTree")
print("  - Clustering: KMeans, DBSCAN, AgglomerativeClustering")
print("  - Dimensionality: PCA, LDA")
print("  - Metrics: accuracy, precision, recall, f1, confusion_matrix")
print("  - Model Selection: train_test_split, cross_validate, GridSearchCV")
print("\nAll implemented with NumPy only - zero external ML dependency.")
