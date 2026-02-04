# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
Built-in datasets for Neurova.

Provides easy access to bundled datasets for testing and examples.

Usage:
    from neurova.datasets import load_iris, load_titanic, load_cifar10
    
    # Load tabular data
    df = load_iris()
    
    # Load image datasets
    train, test = load_cifar10()
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Data directory path
_DATA_DIR = Path(__file__).resolve().parent


def _get_data_path(subdir: str, filename: str = "") -> Path:
    """Get path to data file or directory."""
    path = _DATA_DIR / subdir
    if filename:
        path = path / filename
    return path


# Tabular Datasets

def load_iris() -> Any:
    """Load the Iris flower dataset.
    
    Returns:
        pandas DataFrame with sepal/petal measurements and species.
        
    Features:
        - sepal_length, sepal_width, petal_length, petal_width
        - species (setosa, versicolor, virginica)
    """
    try:
        import pandas as pd
        path = _get_data_path("tabular", "iris.csv")
        if path.exists():
            return pd.read_csv(path)
        raise FileNotFoundError(f"Iris dataset not found at {path}")
    except ImportError:
        raise ImportError("pandas is required: pip install pandas")


def load_titanic() -> Any:
    """Load the Titanic passenger dataset.
    
    Returns:
        pandas DataFrame with passenger information.
        
    Features:
        - PassengerId, Survived, Pclass, Name, Sex, Age
        - SibSp, Parch, Ticket, Fare, Cabin, Embarked
    """
    try:
        import pandas as pd
        path = _get_data_path("tabular", "titanic.csv")
        if path.exists():
            return pd.read_csv(path)
        raise FileNotFoundError(f"Titanic dataset not found at {path}")
    except ImportError:
        raise ImportError("pandas is required: pip install pandas")


def load_boston_housing() -> Any:
    """Load the Boston Housing dataset.
    
    Returns:
        pandas DataFrame with housing features and prices.
        
    Target: medv (median home value)
    """
    try:
        import pandas as pd
        path = _get_data_path("tabular", "boston-housing.csv")
        if path.exists():
            return pd.read_csv(path)
        raise FileNotFoundError(f"Boston Housing dataset not found at {path}")
    except ImportError:
        raise ImportError("pandas is required: pip install pandas")


def load_diabetes() -> Any:
    """Load the Pima Indians Diabetes dataset.
    
    Returns:
        pandas DataFrame with medical measurements.
        
    Target: Outcome (0 or 1)
    """
    try:
        import pandas as pd
        path = _get_data_path("tabular", "diabetes.csv")
        if path.exists():
            df = pd.read_csv(path, header=None)
            df.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                         'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome']
            return df
        raise FileNotFoundError(f"Diabetes dataset not found at {path}")
    except ImportError:
        raise ImportError("pandas is required: pip install pandas")


def load_wine() -> Any:
    """Load the Wine/Tips dataset.
    
    Returns:
        pandas DataFrame.
    """
    try:
        import pandas as pd
        path = _get_data_path("tabular", "wine.csv")
        if path.exists():
            return pd.read_csv(path)
        raise FileNotFoundError(f"Wine dataset not found at {path}")
    except ImportError:
        raise ImportError("pandas is required: pip install pandas")


# Time Series Datasets

def load_air_passengers() -> Any:
    """Load the Air Passengers time series dataset.
    
    Returns:
        pandas DataFrame with monthly airline passenger numbers (1949-1960).
    """
    try:
        import pandas as pd
        path = _get_data_path("timeseries", "air-passengers.csv")
        if path.exists():
            df = pd.read_csv(path)
            return df
        raise FileNotFoundError(f"Air Passengers dataset not found at {path}")
    except ImportError:
        raise ImportError("pandas is required: pip install pandas")


def load_daily_temperatures() -> Any:
    """Load the Daily Minimum Temperatures dataset.
    
    Returns:
        pandas DataFrame with daily temperature readings.
    """
    try:
        import pandas as pd
        path = _get_data_path("timeseries", "daily-temperatures.csv")
        if path.exists():
            return pd.read_csv(path)
        raise FileNotFoundError(f"Daily Temperatures dataset not found at {path}")
    except ImportError:
        raise ImportError("pandas is required: pip install pandas")


def load_sunspots() -> Any:
    """Load the Monthly Sunspots dataset.
    
    Returns:
        pandas DataFrame with monthly sunspot counts.
    """
    try:
        import pandas as pd
        path = _get_data_path("timeseries", "sunspots.csv")
        if path.exists():
            return pd.read_csv(path)
        raise FileNotFoundError(f"Sunspots dataset not found at {path}")
    except ImportError:
        raise ImportError("pandas is required: pip install pandas")


# Clustering Datasets

def load_penguins() -> Any:
    """Load the Palmer Penguins dataset.
    
    Returns:
        pandas DataFrame with penguin measurements.
        
    Features:
        - species, island, bill_length_mm, bill_depth_mm
        - flipper_length_mm, body_mass_g, sex
    """
    try:
        import pandas as pd
        path = _get_data_path("clustering", "penguins.csv")
        if path.exists():
            return pd.read_csv(path)
        raise FileNotFoundError(f"Penguins dataset not found at {path}")
    except ImportError:
        raise ImportError("pandas is required: pip install pandas")


def load_mall_customers() -> Any:
    """Load the Mall Customers dataset for clustering.
    
    Returns:
        pandas DataFrame with customer information.
        
    Features:
        - CustomerID, Gender, Age, Annual Income, Spending Score
    """
    try:
        import pandas as pd
        path = _get_data_path("clustering", "mall-customers.csv")
        if path.exists():
            return pd.read_csv(path)
        raise FileNotFoundError(f"Mall Customers dataset not found at {path}")
    except ImportError:
        raise ImportError("pandas is required: pip install pandas")


# Image Datasets

def load_sample_image(name: str = "lena") -> np.ndarray:
    """Load a sample image for testing.
    
    Args:
        name: Image name ('lena', 'baboon', 'fruits', 'building', 
              'chessboard', 'sudoku')
    
    Returns:
        numpy array (H, W, C) in BGR format.
    """
    try:
        import cv2
    except ImportError:
        try:
            from neurova import nvc as cv2
        except ImportError:
            raise ImportError("OpenCV or neurova.nvc is required")
    
    # Try common extensions
    for ext in ['.png', '.jpg', '.jpeg']:
        path = _get_data_path("sample-images", f"{name}{ext}")
        if path.exists():
            img = cv2.imread(str(path))
            if img is not None:
                return img
    
    raise FileNotFoundError(f"Sample image '{name}' not found")


def get_sample_images() -> List[str]:
    """Get list of available sample images.
    
    Returns:
        List of image names (without extension).
    """
    path = _get_data_path("sample-images")
    if not path.exists():
        return []
    
    images = []
    for f in path.iterdir():
        if f.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            images.append(f.stem)
    return sorted(images)


def load_fashion_mnist() -> Tuple[Tuple[np.ndarray, np.ndarray], 
                                   Tuple[np.ndarray, np.ndarray]]:
    """Load the Fashion-MNIST dataset.
    
    Returns:
        ((train_images, train_labels), (test_images, test_labels))
        
        Images: (N, 28, 28) uint8
        Labels: (N,) uint8 (0-9)
        
    Classes:
        0: T-shirt/top, 1: Trouser, 2: Pullover, 3: Dress, 4: Coat
        5: Sandal, 6: Shirt, 7: Sneaker, 8: Bag, 9: Ankle boot
    """
    import gzip
    
    def load_images(path: Path) -> np.ndarray:
        with gzip.open(path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28, 28)
    
    def load_labels(path: Path) -> np.ndarray:
        with gzip.open(path, 'rb') as f:
            return np.frombuffer(f.read(), np.uint8, offset=8)
    
    base = _get_data_path("fashion-mnist", "data/fashion")
    
    train_images = load_images(base / "train-images-idx3-ubyte.gz")
    train_labels = load_labels(base / "train-labels-idx1-ubyte.gz")
    test_images = load_images(base / "t10k-images-idx3-ubyte.gz")
    test_labels = load_labels(base / "t10k-labels-idx1-ubyte.gz")
    
    return (train_images, train_labels), (test_images, test_labels)


# Cascade Paths

def get_haarcascade(name: str) -> str:
    """Get path to a Haar cascade classifier.
    
    Args:
        name: Cascade name (e.g., 'frontalface_default', 'eye')
              Will auto-add 'haarcascade_' prefix if missing.
    
    Returns:
        Full path to the cascade XML file.
    """
    if not name.startswith("haarcascade_"):
        name = f"haarcascade_{name}"
    if not name.endswith(".xml"):
        name = f"{name}.xml"
    
    path = _get_data_path("haarcascades", name)
    if path.exists():
        return str(path)
    raise FileNotFoundError(f"Haar cascade '{name}' not found")


def get_lbpcascade(name: str) -> str:
    """Get path to an LBP cascade classifier.
    
    Args:
        name: Cascade name (e.g., 'frontalface', 'frontalcatface')
    
    Returns:
        Full path to the cascade XML file.
    """
    if not name.startswith("lbpcascade_"):
        name = f"lbpcascade_{name}"
    if not name.endswith(".xml"):
        name = f"{name}.xml"
    
    path = _get_data_path("lbpcascades", name)
    if path.exists():
        return str(path)
    raise FileNotFoundError(f"LBP cascade '{name}' not found")


def get_hogcascade(name: str = "pedestrians") -> str:
    """Get path to a HOG cascade classifier.
    
    Args:
        name: Cascade name (default: 'pedestrians')
    
    Returns:
        Full path to the cascade XML file.
    """
    if not name.startswith("hogcascade_"):
        name = f"hogcascade_{name}"
    if not name.endswith(".xml"):
        name = f"{name}.xml"
    
    path = _get_data_path("hogcascades", name)
    if path.exists():
        return str(path)
    raise FileNotFoundError(f"HOG cascade '{name}' not found")


# Dataset Info

def list_datasets() -> Dict[str, List[str]]:
    """List all available datasets.
    
    Returns:
        Dictionary with categories and dataset names.
    """
    datasets = {
        "tabular": [],
        "timeseries": [],
        "clustering": [],
        "images": [],
        "cascades": {
            "haar": [],
            "lbp": [],
            "hog": [],
        }
    }
    
    # Tabular
    tabular_path = _get_data_path("tabular")
    if tabular_path.exists():
        for f in tabular_path.glob("*.csv"):
            datasets["tabular"].append(f.stem)
    
    # Time series
    ts_path = _get_data_path("timeseries")
    if ts_path.exists():
        for f in ts_path.glob("*.csv"):
            datasets["timeseries"].append(f.stem)
    
    # Clustering
    cluster_path = _get_data_path("clustering")
    if cluster_path.exists():
        for f in cluster_path.glob("*.csv"):
            datasets["clustering"].append(f.stem)
    
    # Images
    datasets["images"] = get_sample_images()
    
    # Cascades
    haar_path = _get_data_path("haarcascades")
    if haar_path.exists():
        for f in haar_path.glob("*.xml"):
            datasets["cascades"]["haar"].append(f.stem.replace("haarcascade_", ""))
    
    lbp_path = _get_data_path("lbpcascades")
    if lbp_path.exists():
        for f in lbp_path.glob("*.xml"):
            datasets["cascades"]["lbp"].append(f.stem.replace("lbpcascade_", ""))
    
    hog_path = _get_data_path("hogcascades")
    if hog_path.exists():
        for f in hog_path.glob("*.xml"):
            datasets["cascades"]["hog"].append(f.stem.replace("hogcascade_", ""))
    
    return datasets


__all__ = [
    # Tabular
    "load_iris",
    "load_titanic",
    "load_boston_housing",
    "load_diabetes",
    "load_wine",
    # Time Series
    "load_air_passengers",
    "load_daily_temperatures",
    "load_sunspots",
    # Clustering
    "load_penguins",
    "load_mall_customers",
    # Images
    "load_sample_image",
    "get_sample_images",
    "load_fashion_mnist",
    # Cascades
    "get_haarcascade",
    "get_lbpcascade",
    "get_hogcascade",
    # Info
    "list_datasets",
]
