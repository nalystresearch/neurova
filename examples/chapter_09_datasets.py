# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Chapter 9: Dataset Loading & Management


This chapter covers:
- Loading built-in datasets
- Tabular data processing
- Image datasets
- Time series data
- Custom dataset creation
- Data augmentation

Using Neurova's data management tools!

Author: Neurova Team
"""

import numpy as np
from pathlib import Path

print("")
print("Chapter 9: Dataset Loading & Management")
print("")

import neurova as nv
from neurova import datasets

# get data directory
DATA_DIR = Path(__file__).parent.parent / "neurova" / "data"

# 9.1 available datasets overview
print(f"\n9.1 Available Datasets")

print("\n    Data directory contents:")
for item in sorted(DATA_DIR.iterdir()):
    if item.is_dir():
        count = len(list(item.glob('*')))
        print(f"       {item.name}/ ({count} items)")
    else:
        size = item.stat().st_size / 1024
        print(f"       {item.name} ({size:.1f} KB)")

# 9.2 tabular datasets
print(f"\n9.2 Tabular Datasets")

# list tabular datasets
tabular_dir = DATA_DIR / "tabular"
print(f"\n    Tabular datasets in {tabular_dir.name}/:")
for csv_file in sorted(tabular_dir.glob("*.csv")):
    size = csv_file.stat().st_size / 1024
    print(f"      - {csv_file.name} ({size:.1f} KB)")

# 9.3 iris dataset
print(f"\n9.3 Iris Dataset")

iris = datasets.load_iris()

if isinstance(iris, dict):
    X = iris.get('data') or iris.get('X')
    y = iris.get('target') or iris.get('y')
    feature_names = iris.get('feature_names', [])
    target_names = iris.get('target_names', [])
    
    print(f"    Features: {X.shape if X is not None else 'N/A'}")
    print(f"    Targets: {y.shape if y is not None else 'N/A'}")
    print(f"    Feature names: {feature_names[:4] if feature_names else 'N/A'}")
    print(f"    Target names: {target_names if target_names else 'N/A'}")
    
    if X is not None:
        print(f"\n    Sample data (first 3 rows):")
        for i in range(min(3, len(X))):
            print(f"      {X[i]} -> {y[i] if y is not None else '?'}")

# 9.4 diabetes dataset
print(f"\n9.4 Diabetes Dataset")

diabetes = datasets.load_diabetes()

if isinstance(diabetes, dict):
    X = diabetes.get('data') or diabetes.get('X')
    y = diabetes.get('target') or diabetes.get('y')
    
    print(f"    Features: {X.shape if X is not None else 'N/A'}")
    print(f"    Targets: {y.shape if y is not None else 'N/A'}")
    
    if X is not None:
        print(f"    Feature stats:")
        print(f"      Mean: {X.mean(axis=0)[:5]}")
        print(f"      Std: {X.std(axis=0)[:5]}")

# 9.5 boston housing dataset
print(f"\n9.5 Boston Housing Dataset")

boston = datasets.load_boston_housing()

if isinstance(boston, dict):
    X = boston.get('data') or boston.get('X')
    y = boston.get('target') or boston.get('y')
    
    print(f"    Features: {X.shape if X is not None else 'N/A'}")
    print(f"    Targets: {y.shape if y is not None else 'N/A'}")
    
    if y is not None:
        print(f"    Price range: ${y.min()*1000:.0f} - ${y.max()*1000:.0f}")
        print(f"    Median price: ${y.mean()*1000:.0f}")

# 9.6 titanic dataset
print(f"\n9.6 Titanic Dataset")

titanic = datasets.load_titanic()

if isinstance(titanic, dict):
    X = titanic.get('data') or titanic.get('X')
    y = titanic.get('target') or titanic.get('y')
    feature_names = titanic.get('feature_names', [])
    
    print(f"    Features: {X.shape if X is not None else 'N/A'}")
    print(f"    Targets: {y.shape if y is not None else 'N/A'}")
    
    if y is not None:
        survived = np.sum(y == 1)
        total = len(y)
        print(f"    Survival rate: {survived}/{total} ({survived/total*100:.1f}%)")

# 9.7 wine dataset
print(f"\n9.7 Wine Dataset")

wine = datasets.load_wine()

if isinstance(wine, dict):
    X = wine.get('data') or wine.get('X')
    y = wine.get('target') or wine.get('y')
    
    print(f"    Features: {X.shape if X is not None else 'N/A'}")
    print(f"    Targets: {y.shape if y is not None else 'N/A'}")
    
    if y is not None:
        unique, counts = np.unique(y, return_counts=True)
        print(f"    Classes: {dict(zip(unique, counts))}")

# 9.8 time series datasets
print(f"\n9.8 Time Series Datasets")

timeseries_dir = DATA_DIR / "timeseries"
if timeseries_dir.exists():
    print(f"\n    Time series datasets:")
    for csv_file in sorted(timeseries_dir.glob("*.csv")):
        print(f"      - {csv_file.name}")

# air passengers
air_passengers = datasets.load_air_passengers()
if isinstance(air_passengers, dict):
    data = air_passengers.get('data')
    if data is not None:
        print(f"\n    Air Passengers:")
        print(f"      Shape: {data.shape}")
        print(f"      Range: {data.min()} - {data.max()}")

# 9.9 clustering datasets
print(f"\n9.9 Clustering Datasets")

clustering_dir = DATA_DIR / "clustering"
if clustering_dir.exists():
    print(f"\n    Clustering datasets:")
    for csv_file in sorted(clustering_dir.glob("*.csv")):
        print(f"      - {csv_file.name}")

# mall customers
mall = datasets.load_mall_customers()
if isinstance(mall, dict):
    X = mall.get('data') or mall.get('X')
    if X is not None:
        print(f"\n    Mall Customers:")
        print(f"      Features: {X.shape}")

# penguins
try:
    penguins = datasets.load_penguins()
    if isinstance(penguins, dict):
        X = penguins.get('data') or penguins.get('X')
        if X is not None:
            print(f"\n    Penguins:")
            print(f"      Features: {X.shape}")
except:
    pass

# 9.10 fashion-mnist dataset
print(f"\n9.10 Fashion-MNIST Dataset")

fashion_dir = DATA_DIR / "fashion-mnist"
if fashion_dir.exists():
    print(f"    Fashion-MNIST directory exists")
    
# load if available
    try:
        fashion = datasets.load_fashion_mnist()
        if isinstance(fashion, dict):
            X_train = fashion.get('X_train')
            y_train = fashion.get('y_train')
            X_test = fashion.get('X_test')
            y_test = fashion.get('y_test')
            
            if X_train is not None:
                print(f"      Training: {X_train.shape}")
                print(f"      Testing: {X_test.shape if X_test is not None else 'N/A'}")
                
# class names
                class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
                print(f"      Classes: {class_names}")
    except Exception as e:
        print(f"      (Load function: {e})")

# 9.11 movielens dataset
print(f"\n9.11 MovieLens Dataset")

movielens_dir = DATA_DIR / "movielens-100k"
if movielens_dir.exists():
    print(f"    MovieLens 100K directory exists")
    
# list files
    for f in sorted(movielens_dir.iterdir())[:5]:
        print(f"      - {f.name}")
    
# load ratings
    try:
        movielens = datasets.load_movielens()
        if isinstance(movielens, dict):
            ratings = movielens.get('ratings')
            if ratings is not None:
                print(f"\n      Ratings: {ratings.shape}")
                print(f"      Users: {ratings[:, 0].max():.0f}")
                print(f"      Movies: {ratings[:, 1].max():.0f}")
    except Exception as e:
        print(f"      (Load function: {e})")

# 9.12 haar cascades (detection data)
print(f"\n9.12 Haar Cascades")

haar_dir = DATA_DIR / "haarcascades"
if haar_dir.exists():
    cascades = list(haar_dir.glob("*.xml"))
    print(f"    Available cascades: {len(cascades)}")
    for cascade in sorted(cascades)[:5]:
        print(f"      - {cascade.name}")
    if len(cascades) > 5:
        print(f"      ... and {len(cascades) - 5} more")

# 9.13 creating custom datasets
print(f"\n9.13 Creating Custom Datasets")

class CustomDataset:
    """Custom dataset class for Neurova."""
    
    def __init__(self, X, y=None, transform=None):
        self.X = np.array(X)
        self.y = np.array(y) if y is not None else None
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform:
            x = self.transform(x)
        
        if self.y is not None:
            return x, self.y[idx]
        return x
    
    def get_batch(self, indices):
        """Get a batch of samples."""
        X_batch = self.X[indices]
        if self.transform:
            X_batch = np.array([self.transform(x) for x in X_batch])
        
        if self.y is not None:
            return X_batch, self.y[indices]
        return X_batch

# create custom dataset
X_custom = np.random.randn(100, 10)
y_custom = np.random.randint(0, 3, 100)

custom_ds = CustomDataset(X_custom, y_custom)
print(f"    Custom dataset: {len(custom_ds)} samples")

# access samples
sample_x, sample_y = custom_ds[0]
print(f"    Sample shape: {sample_x.shape}, label: {sample_y}")

# get batch
batch_x, batch_y = custom_ds.get_batch(np.arange(8))
print(f"    Batch shape: {batch_x.shape}, labels: {batch_y}")

# 9.14 data augmentation
print(f"\n9.14 Data Augmentation")

class DataAugmentation:
    """Data augmentation transforms."""
    
    @staticmethod
    def random_flip_horizontal(image):
        """Randomly flip image horizontally."""
        if np.random.rand() > 0.5:
            return np.fliplr(image)
        return image
    
    @staticmethod
    def random_flip_vertical(image):
        """Randomly flip image vertically."""
        if np.random.rand() > 0.5:
            return np.flipud(image)
        return image
    
    @staticmethod
    def random_rotation(image, max_angle=15):
        """Random rotation (simplified - 90 degree increments)."""
        k = np.random.randint(0, 4)
        return np.rot90(image, k)
    
    @staticmethod
    def random_noise(image, sigma=0.02):
        """Add random Gaussian noise."""
        noise = np.random.randn(*image.shape) * sigma
        return np.clip(image + noise, 0, 1)
    
    @staticmethod
    def random_brightness(image, factor_range=(0.8, 1.2)):
        """Random brightness adjustment."""
        factor = np.random.uniform(*factor_range)
        return np.clip(image * factor, 0, 1)
    
    @staticmethod
    def compose(image, transforms):
        """Apply multiple transforms."""
        for transform in transforms:
            image = transform(image)
        return image

# create sample image
sample_image = np.random.rand(28, 28)

# apply augmentations
flipped = DataAugmentation.random_flip_horizontal(sample_image)
rotated = DataAugmentation.random_rotation(sample_image)
noisy = DataAugmentation.random_noise(sample_image)
bright = DataAugmentation.random_brightness(sample_image)

print(f"    Original shape: {sample_image.shape}")
print(f"    Flipped shape: {flipped.shape}")
print(f"    Rotated shape: {rotated.shape}")
print(f"    Noisy range: [{noisy.min():.3f}, {noisy.max():.3f}]")
print(f"    Brightness range: [{bright.min():.3f}, {bright.max():.3f}]")

# compose transforms
composed = DataAugmentation.compose(sample_image, [
    DataAugmentation.random_flip_horizontal,
    DataAugmentation.random_noise,
    DataAugmentation.random_brightness
])
print(f"    Composed shape: {composed.shape}")

# 9.15 data loaders
print(f"\n9.15 Data Loaders")

class DataLoader:
    """Mini-batch data loader."""
    
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        n = len(self.dataset)
        indices = np.arange(n)
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch_indices = indices[start:end]
            yield self.dataset.get_batch(batch_indices)

# create data loader
loader = DataLoader(custom_ds, batch_size=16, shuffle=True)
print(f"    DataLoader: {len(loader)} batches of size 16")

# iterate through loader
batch_count = 0
for X_batch, y_batch in loader:
    batch_count += 1
print(f"    Iterated through {batch_count} batches")

# 9.16 train/validation/test split
print(f"\n9.16 Train/Validation/Test Split")

def train_val_test_split(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=None):
    """Split data into train, validation, and test sets."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    if random_state is not None:
        np.random.seed(random_state)
    
    n = len(X)
    indices = np.random.permutation(n)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    return (
        X[train_idx], y[train_idx],
        X[val_idx], y[val_idx],
        X[test_idx], y[test_idx]
    )

# split data
X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
    X_custom, y_custom, 
    train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
    random_state=42
)

print(f"    Total: {len(X_custom)}")
print(f"    Train: {len(X_train)} ({len(X_train)/len(X_custom)*100:.0f}%)")
print(f"    Val: {len(X_val)} ({len(X_val)/len(X_custom)*100:.0f}%)")
print(f"    Test: {len(X_test)} ({len(X_test)/len(X_custom)*100:.0f}%)")

# summary
print("\n" + "=" * 60)
print("Chapter 9 Summary:")
print("   Explored available datasets in neurova/data/")
print("   Loaded tabular datasets (iris, diabetes, boston, titanic, wine)")
print("   Loaded time series data (air passengers)")
print("   Loaded clustering data (mall customers, penguins)")
print("   Loaded image datasets (fashion-mnist)")
print("   Loaded recommendation data (movielens)")
print("   Found Haar cascades for detection")
print("   Created custom dataset class")
print("   Implemented data augmentation")
print("   Built data loaders for batching")
print("   Split data into train/val/test sets")
print("")
