# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
Data files and resources for Neurova

Provides paths to bundled data files like cascade classifiers.
Compatible with nvc.data.haarcascades usage pattern.
"""

from __future__ import annotations

import os
from pathlib import Path


def _find_data_path() -> Path:
    """Find the data directory with cascade files."""
    here = Path(__file__).resolve().parent
    
    # Check various possible locations
    candidates = [
        here / "haarcascades",                    # Inside package
        here.parent.parent / "data" / "haarcascades",  # Project data folder
        here.parents[2] / "example" / "data" / "haarcascades",  # Example folder
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return candidate
    
    return here


def _find_cascade_dir(cascade_type: str = "haarcascades") -> str:
    """Find directory for cascade classifiers.
    
    Args:
        cascade_type: Type of cascades ("haarcascades", "hogcascades", "lbpcascades")
    
    Returns:
        Path to cascade directory with trailing separator
    """
    here = Path(__file__).resolve().parent
    
    candidates = [
        here / cascade_type,
        here.parent.parent / "data" / cascade_type,
        here.parents[2] / "example" / "data" / cascade_type,
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return str(candidate) + os.sep
    
    return ""


# Neurova nvc.data.haarcascades compatible path
haarcascades: str = _find_cascade_dir("haarcascades")
hogcascades: str = _find_cascade_dir("hogcascades")
lbpcascades: str = _find_cascade_dir("lbpcascades")


# Import data loading utilities
from neurova.data.dataloader import (
    Dataset,
    IterableDataset,
    TensorDataset,
    ConcatDataset,
    Subset,
    ChainDataset,
    random_split,
    Sampler,
    SequentialSampler,
    RandomSampler,
    SubsetRandomSampler,
    WeightedRandomSampler,
    BatchSampler,
    DistributedSampler,
    DataLoader,
    default_collate,
    default_convert,
    ImageFolder,
)


# Import dataset loading functions
from neurova.data.datasets import (
    # Tabular
    load_iris,
    load_titanic,
    load_boston_housing,
    load_diabetes,
    load_wine,
    # Time Series
    load_air_passengers,
    load_daily_temperatures,
    load_sunspots,
    # Clustering
    load_penguins,
    load_mall_customers,
    # Images
    load_sample_image,
    get_sample_images,
    load_fashion_mnist,
    # Cascades
    get_haarcascade,
    get_lbpcascade,
    get_hogcascade,
    # Info
    list_datasets,
)


__all__ = [
    # Cascade paths
    "haarcascades",
    "hogcascades",
    "lbpcascades",
    # Dataset classes
    "Dataset",
    "IterableDataset",
    "TensorDataset",
    "ConcatDataset",
    "Subset",
    "ChainDataset",
    "random_split",
    # Samplers
    "Sampler",
    "SequentialSampler",
    "RandomSampler",
    "SubsetRandomSampler",
    "WeightedRandomSampler",
    "BatchSampler",
    "DistributedSampler",
    # DataLoader
    "DataLoader",
    "default_collate",
    "default_convert",
    # Utilities
    "ImageFolder",
    # Dataset loaders
    "load_iris",
    "load_titanic",
    "load_boston_housing",
    "load_diabetes",
    "load_wine",
    "load_air_passengers",
    "load_daily_temperatures",
    "load_sunspots",
    "load_penguins",
    "load_mall_customers",
    "load_sample_image",
    "get_sample_images",
    "load_fashion_mnist",
    "get_haarcascade",
    "get_lbpcascade",
    "get_hogcascade",
    "list_datasets",
]
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.