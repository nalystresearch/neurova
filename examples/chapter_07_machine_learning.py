# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
Chapter 7: Machine Learning with Neurova
==========================================

This chapter covers:
- Loading tabular datasets
- Data preprocessing
- Classification algorithms
- Regression algorithms
- Clustering algorithms
- Model evaluation metrics

Using Neurova's ML implementations!

Author: Neurova Team
"""

import numpy as np
from pathlib import Path

print("=" * 60)
print("Chapter 7: Machine Learning with Neurova")
print("=" * 60)

import neurova as nv
from neurova import datasets

# get data directory
DATA_DIR = Path(__file__).parent.parent / "neurova" / "data"
TABULAR_DIR = DATA_DIR / "tabular"
CLUSTERING_DIR = DATA_DIR / "clustering"

# 7.1 loading datasets
print(f"\n7.1 Loading Datasets")

# load iris dataset
iris_data = datasets.load_iris()
print(f"    Iris dataset loaded: {type(iris_data)}")
if isinstance(iris_data, dict):
    print(f"      Keys: {list(iris_data.keys())}")
    X_iris = iris_data.get('data') or iris_data.get('X')
    y_iris = iris_data.get('target') or iris_data.get('y')
    if X_iris is not None:
        print(f"      Features shape: {X_iris.shape}")
        print(f"      Target shape: {y_iris.shape}")

# load diabetes dataset
diabetes_data = datasets.load_diabetes()
print(f"\n    Diabetes dataset loaded: {type(diabetes_data)}")

# load boston housing
boston_data = datasets.load_boston_housing()
print(f"    Boston Housing loaded: {type(boston_data)}")

# load titanic
titanic_data = datasets.load_titanic()
print(f"    Titanic loaded: {type(titanic_data)}")

# 7.2 data preprocessing
print(f"\n7.2 Data Preprocessing")

from neurova.ml import preprocessing

# create sample data
X_sample = np.array([
    [1.0, 100, 0.5],
    [2.0, 200, 1.0],
    [3.0, 300, 1.5],
    [4.0, 400, 2.0],
    [5.0, 500, 2.5]
])

print(f"    Original data:\n{X_sample}")

# standardization using standardscaler
scaler = preprocessing.StandardScaler()
X_standardized = scaler.fit_transform(X_sample)
print(f"\n    Standardized (mean=0, std=1):")
print(f"      Mean: {X_standardized.mean(axis=0)}")
print(f"      Std: {X_standardized.std(axis=0)}")

# min-max normalization using minmaxscaler
minmax = preprocessing.MinMaxScaler()
X_normalized = minmax.fit_transform(X_sample)
print(f"\n    Normalized [0, 1]:")
print(f"      Min: {X_normalized.min(axis=0)}")
print(f"      Max: {X_normalized.max(axis=0)}")

# 7.3 train-test split
print(f"\n7.3 Train-Test Split")

from neurova.ml import train_test_split

# use iris dataset from neurova
iris_df = datasets.load_iris()

# extract features and labels from dataframe
if hasattr(iris_df, 'values'):
# pandas dataframe - get numeric columns for features
    X = iris_df.iloc[:, :-1].values.astype(np.float64)
    y = iris_df.iloc[:, -1].values
# encode species names to integers if needed
    if isinstance(y[0], str):
        unique_species = list(set(y))
        y = np.array([unique_species.index(s) for s in y])
else:
# dictionary format
    X = np.array(iris_df.get('data', iris_df.get('X')))
    y = np.array(iris_df.get('target', iris_df.get('y')))

print(f"    Using Iris dataset from Neurova")
print(f"    Features: {X.shape}, Classes: {len(np.unique(y))}")

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"    Original: X={X.shape}, y={y.shape}")
print(f"    Train: X={X_train.shape}, y={y_train.shape}")
print(f"    Test: X={X_test.shape}, y={y_test.shape}")

# 7.4 k-nearest neighbors classifier
print(f"\n7.4 K-Nearest Neighbors (KNN)")

from neurova.ml import KNearestNeighbors

# create classifier
knn = KNearestNeighbors(n_neighbors=5)

# train
knn.fit(X_train, y_train)
print(f"    Trained KNN with k={knn.n_neighbors}")

# predict
y_pred = knn.predict(X_test)
print(f"    Predictions: {y_pred[:10]}")

# accuracy
accuracy = np.mean(y_pred == y_test) * 100
print(f"    Accuracy: {accuracy:.2f}%")

# 7.5 decision tree classifier
print(f"\n7.5 Decision Tree Classifier")

from neurova.ml import DecisionTreeClassifier

# create classifier
dt = DecisionTreeClassifier(max_depth=5)

# train
dt.fit(X_train, y_train)
print(f"    Trained Decision Tree with max_depth={dt.max_depth}")

# predict
dt_pred = dt.predict(X_test)
dt_accuracy = np.mean(dt_pred == y_test) * 100
print(f"    Accuracy: {dt_accuracy:.2f}%")

# 7.6 naive bayes classifier
print(f"\n7.6 Naive Bayes Classifier")

from neurova.ml import NaiveBayes

# create classifier
nb = NaiveBayes()

# train
nb.fit(X_train, y_train)
print(f"    Trained Gaussian Naive Bayes")

# predict
nb_pred = nb.predict(X_test)
nb_accuracy = np.mean(nb_pred == y_test) * 100
print(f"    Accuracy: {nb_accuracy:.2f}%")

# 7.7 logistic regression
print(f"\n7.7 Logistic Regression")

from neurova.ml import LogisticRegression

# Create classifier (for binary classification, use 2 classes)
X_binary = X[y != 2]
y_binary = y[y != 2]
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_binary, y_binary, test_size=0.2, random_state=42
)

lr = LogisticRegression(learning_rate=0.01, n_iterations=100)

# train
lr.fit(X_train_b, y_train_b)
print(f"    Trained Logistic Regression")

# predict
lr_pred = lr.predict(X_test_b)
lr_accuracy = np.mean(lr_pred == y_test_b) * 100
print(f"    Accuracy: {lr_accuracy:.2f}%")

# 7.8 linear regression
print(f"\n7.8 Linear Regression")

from neurova.ml import LinearRegression

# use boston housing dataset from neurova for regression
boston_df = datasets.load_boston_housing()

if hasattr(boston_df, 'values'):
    # Pandas DataFrame - target is usually last column 'medv'
    X_reg = boston_df.iloc[:, :-1].values.astype(np.float64)
    y_reg = boston_df.iloc[:, -1].values.astype(np.float64)
else:
    X_reg = np.array(boston_df.get('data', boston_df.get('X')))
    y_reg = np.array(boston_df.get('target', boston_df.get('y')))

print(f"    Using Boston Housing dataset from Neurova")
print(f"    Features: {X_reg.shape}")

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# create and train
lin_reg = LinearRegression()
lin_reg.fit(X_train_r, y_train_r)
print(f"    Trained Linear Regression")
print(f"    Coefficients (first 5): {lin_reg.coef_[:5]}")

# predict
y_pred_r = lin_reg.predict(X_test_r)

# MSE
mse = np.mean((y_pred_r - y_test_r) ** 2)
print(f"    MSE: {mse:.4f}")

# R² score
ss_res = np.sum((y_test_r - y_pred_r) ** 2)
ss_tot = np.sum((y_test_r - y_test_r.mean()) ** 2)
r2 = 1 - ss_res / ss_tot
print(f"    R² Score: {r2:.4f}")

# 7.9 k-means clustering
print(f"\n7.9 K-Means Clustering")

from neurova.ml import KMeans

# use mall customers dataset from neurova for clustering
try:
    mall_df = datasets.load_mall_customers()
    if hasattr(mall_df, 'values'):
# use annual income and spending score columns for clustering
        # These are typically columns 3 and 4 (0-indexed)
        X_cluster = mall_df.iloc[:, 3:5].values.astype(np.float64)
        print(f"    Using Mall Customers dataset from Neurova")
    else:
        raise ValueError("Dataset not in expected format")
except Exception as e:
    # Fallback: try Penguins dataset
    try:
        penguins_df = datasets.load_penguins()
        if hasattr(penguins_df, 'values'):
            # Use numeric columns only, drop NaN
            numeric_cols = penguins_df.select_dtypes(include=[np.number])
            X_cluster = numeric_cols.dropna().values[:90, :2]  # First 2 numeric columns
            print(f"    Using Penguins dataset from Neurova")
        else:
            raise ValueError("Dataset not in expected format")
    except:
        # Final fallback: generate synthetic data
        np.random.seed(42)
        cluster1 = np.random.randn(30, 2) + [0, 0]
        cluster2 = np.random.randn(30, 2) + [5, 5]
        cluster3 = np.random.randn(30, 2) + [10, 0]
        X_cluster = np.vstack([cluster1, cluster2, cluster3])
        print(f"    Using synthetic clustering data (datasets not available)")

print(f"    Data shape: {X_cluster.shape}")

# create and fit
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_cluster)

print(f"    K-Means with k={kmeans.n_clusters}")
print(f"    Cluster centers:\n{kmeans.cluster_centers_}")

# predict cluster labels
cluster_labels = kmeans.predict(X_cluster)
print(f"    Cluster distribution: {np.bincount(cluster_labels)}")

# 7.10 hierarchical clustering
print(f"\n7.10 Hierarchical Clustering")

from neurova.ml import AgglomerativeClustering

# create and fit
agg = AgglomerativeClustering(n_clusters=3, linkage='complete')
agg_labels = agg.fit_predict(X_cluster)

print(f"    Agglomerative Clustering with k={agg.n_clusters}")
print(f"    Linkage: {agg.linkage}")
print(f"    Cluster distribution: {np.bincount(agg_labels)}")

# 7.11 dbscan clustering
print(f"\n7.11 DBSCAN Clustering")

from neurova.ml import DBSCAN

# create and fit
dbscan = DBSCAN(eps=1.0, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_cluster)

n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

print(f"    DBSCAN with eps={dbscan.eps}, min_samples={dbscan.min_samples}")
print(f"    Found clusters: {n_clusters}")
print(f"    Noise points: {n_noise}")

# 7.12 principal component analysis (pca)
print(f"\n7.12 Principal Component Analysis (PCA)")

from neurova.ml import PCA

# Use Iris dataset features for PCA (4 features)
# we already have x from iris loaded earlier
print(f"    Using Iris dataset features for dimensionality reduction")
print(f"    Original features: {X.shape[1]} dimensions")

# Apply PCA to reduce to 2 components
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print(f"    Original shape: {X.shape}")
print(f"    Reduced shape: {X_reduced.shape}")
print(f"    Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"    Total variance explained: {sum(pca.explained_variance_ratio_):.4f}")

# 7.13 model evaluation metrics
print(f"\n7.13 Model Evaluation Metrics")

from neurova.ml import metrics

# classification metrics
y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2])
y_pred_eval = np.array([0, 1, 1, 1, 2, 0, 0, 1, 2])

accuracy = metrics.accuracy_score(y_true, y_pred_eval)
print(f"    Accuracy: {accuracy:.4f}")

precision = metrics.precision_score(y_true, y_pred_eval, average='macro')
print(f"    Precision (macro): {precision:.4f}")

recall = metrics.recall_score(y_true, y_pred_eval, average='macro')
print(f"    Recall (macro): {recall:.4f}")

f1 = metrics.f1_score(y_true, y_pred_eval, average='macro')
print(f"    F1-Score (macro): {f1:.4f}")

# confusion matrix
cm = metrics.confusion_matrix(y_true, y_pred_eval)
print(f"    Confusion Matrix:\n{cm}")

# 7.14 regression metrics
print(f"\n7.14 Regression Metrics")

y_true_reg = np.array([3.0, 5.0, 2.5, 7.0])
y_pred_reg = np.array([2.5, 5.0, 4.0, 8.0])

mse = metrics.mean_squared_error(y_true_reg, y_pred_reg)
print(f"    MSE: {mse:.4f}")

rmse = np.sqrt(mse)
print(f"    RMSE: {rmse:.4f}")

mae = metrics.mean_absolute_error(y_true_reg, y_pred_reg)
print(f"    MAE: {mae:.4f}")

r2 = metrics.r2_score(y_true_reg, y_pred_reg)
print(f"    R² Score: {r2:.4f}")

# 7.15 cross-validation
print(f"\n7.15 Cross-Validation")

from neurova.ml import cross_validate

# cross-validate knn
cv_results = cross_validate(knn, X, y, cv=5)
cv_scores = cv_results['test_score']

print(f"    5-Fold CV Scores: {cv_scores}")
print(f"    Mean: {np.mean(cv_scores):.4f}")
print(f"    Std: {np.std(cv_scores):.4f}")

# 7.16 complete ml pipeline
print(f"\n7.16 Complete ML Pipeline")

def ml_pipeline(X, y, model_class, **model_params):
    """
    Complete ML pipeline: preprocess, split, train, evaluate.
    """
    # 1. Preprocess
    scaler = preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # 3. Train
    model = model_class(**model_params)
    model.fit(X_train, y_train)
    
    # 4. Evaluate
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    
    return {
        'model': model,
        'accuracy': accuracy,
        'predictions': y_pred
    }

# run pipeline
result = ml_pipeline(X, y, KNearestNeighbors, n_neighbors=3)
print(f"    Pipeline accuracy: {result['accuracy']:.4f}")

# summary
print("\n" + "=" * 60)
print("Chapter 7 Summary:")
print("   Loaded tabular datasets (iris, diabetes, boston, titanic)")
print("   Preprocessed data (standardization, normalization)")
print("   Trained classifiers (KNN, Decision Tree, Naive Bayes, Logistic)")
print("   Trained regressors (Linear Regression)")
print("   Applied clustering (K-Means, Hierarchical, DBSCAN)")
print("   Used PCA for dimensionality reduction")
print("   Evaluated models with multiple metrics")
print("   Performed cross-validation")
print("   Built complete ML pipeline")
print("=" * 60)
