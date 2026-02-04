# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Test script for all critical Neurova features."""

import numpy as np
np.random.seed(42)

print("="*60)
print("VERIFYING ALL CRITICAL FEATURES ARE IMPLEMENTED")
print("="*60)

# 1. DataLoader/Dataset/Sampler
print("\n[1] DataLoader/Dataset/Sampler...")
from neurova.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler, random_split
X, y = np.random.randn(50, 4), np.random.randint(0, 2, 50)
ds = TensorDataset(X, y)
loader = DataLoader(ds, batch_size=10, shuffle=True)
batch = next(iter(loader))
print(f"    DataLoader: batch shape {batch[0].shape}")

# 2. DecisionTreeRegressor
print("\n[2] DecisionTreeRegressor...")
from neurova.ml import DecisionTreeRegressor
y_reg = X[:, 0] * 2 + X[:, 1] + np.random.randn(50) * 0.1
tree = DecisionTreeRegressor(max_depth=5)
tree.fit(X[:40], y_reg[:40])
print(f"    DecisionTreeRegressor: R²={tree.score(X[40:], y_reg[40:]):.3f}")

# 3. RandomForestRegressor
print("\n[3] RandomForestRegressor...")
from neurova.ml import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=10, max_depth=5)
rf.fit(X[:40], y_reg[:40])
print(f"    RandomForestRegressor: R²={rf.score(X[40:], y_reg[40:]):.3f}")

# 4. GradientBoostingRegressor
print("\n[4] GradientBoostingRegressor...")
from neurova.ml import GradientBoostingRegressor
gb = GradientBoostingRegressor(n_estimators=20, learning_rate=0.1)
gb.fit(X[:40], y_reg[:40])
print(f"    GradientBoostingRegressor: R²={gb.score(X[40:], y_reg[40:]):.3f}")

# 5. Pipeline & ColumnTransformer
print("\n[5] Pipeline & ColumnTransformer...")
from neurova.ml import Pipeline, ColumnTransformer, StandardScaler, LogisticRegression
pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
pipe.fit(X[:40], y[:40])
pred = pipe.predict(X[40:])
print(f"    Pipeline: predictions shape {pred.shape}")
ct = ColumnTransformer([("scale", StandardScaler(), [0, 1]), ("pass", "passthrough", [2, 3])])
Xt = ct.fit_transform(X)
print(f"    ColumnTransformer: output shape {Xt.shape}")

# 6. RandomizedSearchCV
print("\n[6] RandomizedSearchCV...")
from neurova.ml import RandomizedSearchCV, KNearestNeighbors
search = RandomizedSearchCV(KNearestNeighbors, {"n_neighbors": [3,5,7]}, n_iter=2, cv=2)
search.fit(X[:40], y[:40])
print(f"    RandomizedSearchCV: best_params={search.best_params_}")

# 7. StratifiedKFold, GroupKFold
print("\n[7] StratifiedKFold & GroupKFold...")
from neurova.ml import StratifiedKFold, GroupKFold
skf = StratifiedKFold(n_splits=3)
folds = list(skf.split(X, y))
print(f"    StratifiedKFold: {len(folds)} folds")
gkf = GroupKFold(n_splits=3)
groups = np.repeat(np.arange(10), 5)
folds = list(gkf.split(X, y, groups))
print(f"    GroupKFold: {len(folds)} folds")

print("\n" + "="*60)
print("ALL CRITICAL FEATURES VERIFIED!")
print("="*60)
