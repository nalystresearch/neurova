# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
standard-style DataLoader, Dataset, and Sampler implementations.

This module provides a complete data loading pipeline for deep learning,
compatible with standard data loader API.
"""

import numpy as np
from typing import (
    Any, Callable, Dict, Generic, Iterator, List, Optional,
    Sequence, Tuple, TypeVar, Union
)
from abc import ABC, abstractmethod
import threading
from queue import Queue
import multiprocessing as mp

__all__ = [
    # Datasets
    'Dataset', 'IterableDataset', 'TensorDataset', 'ConcatDataset',
    'Subset', 'ChainDataset', 'random_split',
    # Samplers
    'Sampler', 'SequentialSampler', 'RandomSampler', 'SubsetRandomSampler',
    'WeightedRandomSampler', 'BatchSampler', 'DistributedSampler',
    # DataLoader
    'DataLoader',
    # Collate functions
    'default_collate', 'default_convert',
]

T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)


# Dataset Base Classes

class Dataset(ABC, Generic[T_co]):
    """
    Abstract base class for map-style datasets.
    
    All datasets should subclass this class and implement:
        - __getitem__: Returns sample at given index
        - __len__: Returns the size of the dataset
    
    Example:
        >>> class MyDataset(Dataset):
        ...     def __init__(self, data, labels):
        ...         self.data = data
        ...         self.labels = labels
        ...     
        ...     def __getitem__(self, index):
        ...         return self.data[index], self.labels[index]
        ...     
        ...     def __len__(self):
        ...         return len(self.data)
    """
    
    @abstractmethod
    def __getitem__(self, index: int) -> T_co:
        """Return sample at given index."""
        raise NotImplementedError
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the size of the dataset."""
        raise NotImplementedError
    
    def __add__(self, other: 'Dataset[T_co]') -> 'ConcatDataset[T_co]':
        """Concatenate two datasets."""
        return ConcatDataset([self, other])


class IterableDataset(ABC, Generic[T_co]):
    """
    Abstract base class for iterable-style datasets.
    
    Useful for streaming data or when random access is not possible.
    
    Example:
        >>> class MyIterableDataset(IterableDataset):
        ...     def __init__(self, start, end):
        ...         self.start = start
        ...         self.end = end
        ...     
        ...     def __iter__(self):
        ...         for i in range(self.start, self.end):
        ...             yield i
    """
    
    @abstractmethod
    def __iter__(self) -> Iterator[T_co]:
        """Return an iterator over the dataset."""
        raise NotImplementedError
    
    def __add__(self, other: 'IterableDataset[T_co]') -> 'ChainDataset[T_co]':
        """Chain two iterable datasets."""
        return ChainDataset([self, other])


class TensorDataset(Dataset[Tuple[np.ndarray, ...]]):
    """
    Dataset wrapping numpy arrays (tensors).
    
    Each sample will be a tuple of tensors from the same index.
    
    Args:
        *tensors: Arrays to wrap. All must have the same length in first dimension.
    
    Example:
        >>> X = np.random.randn(100, 10)
        >>> y = np.random.randint(0, 2, 100)
        >>> dataset = TensorDataset(X, y)
        >>> x_sample, y_sample = dataset[0]
    """
    
    def __init__(self, *tensors: np.ndarray):
        if len(tensors) == 0:
            raise ValueError("At least one tensor is required")
        
        # Ensure all tensors have the same first dimension
        first_size = len(tensors[0])
        for i, t in enumerate(tensors):
            if len(t) != first_size:
                raise ValueError(
                    f"Size mismatch between tensors: tensor 0 has {first_size} samples, "
                    f"tensor {i} has {len(t)} samples"
                )
        
        self.tensors = tensors
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, ...]:
        return tuple(t[index] for t in self.tensors)
    
    def __len__(self) -> int:
        return len(self.tensors[0])


class ConcatDataset(Dataset[T_co]):
    """
    Concatenate multiple datasets.
    
    Args:
        datasets: List of datasets to concatenate
    
    Example:
        >>> dataset1 = TensorDataset(np.random.randn(100, 10))
        >>> dataset2 = TensorDataset(np.random.randn(50, 10))
        >>> combined = ConcatDataset([dataset1, dataset2])
        >>> len(combined)  # 150
    """
    
    def __init__(self, datasets: Sequence[Dataset[T_co]]):
        if len(datasets) == 0:
            raise ValueError("At least one dataset is required")
        
        self.datasets = list(datasets)
        self.cumulative_sizes = self._cumsum([len(d) for d in self.datasets])
    
    @staticmethod
    def _cumsum(sequence: List[int]) -> List[int]:
        result = []
        total = 0
        for s in sequence:
            total += s
            result.append(total)
        return result
    
    def __getitem__(self, index: int) -> T_co:
        if index < 0:
            if -index > len(self):
                raise IndexError(f"Index {index} out of range")
            index = len(self) + index
        
        # Find which dataset contains this index
        dataset_idx = 0
        for i, size in enumerate(self.cumulative_sizes):
            if index < size:
                dataset_idx = i
                break
        
        if dataset_idx == 0:
            sample_idx = index
        else:
            sample_idx = index - self.cumulative_sizes[dataset_idx - 1]
        
        return self.datasets[dataset_idx][sample_idx]
    
    def __len__(self) -> int:
        return self.cumulative_sizes[-1]


class Subset(Dataset[T_co]):
    """
    Subset of a dataset at specified indices.
    
    Args:
        dataset: The full dataset
        indices: Sequence of indices to include in subset
    
    Example:
        >>> dataset = TensorDataset(np.random.randn(100, 10))
        >>> train_indices = list(range(80))
        >>> train_dataset = Subset(dataset, train_indices)
    """
    
    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]):
        self.dataset = dataset
        self.indices = indices
    
    def __getitem__(self, index: int) -> T_co:
        if isinstance(index, list):
            return [self.dataset[self.indices[i]] for i in index]
        return self.dataset[self.indices[index]]
    
    def __len__(self) -> int:
        return len(self.indices)


class ChainDataset(IterableDataset[T_co]):
    """
    Chain multiple iterable datasets.
    
    Args:
        datasets: Iterable datasets to chain
    """
    
    def __init__(self, datasets: Sequence[IterableDataset[T_co]]):
        self.datasets = list(datasets)
    
    def __iter__(self) -> Iterator[T_co]:
        for dataset in self.datasets:
            yield from dataset


def random_split(
    dataset: Dataset[T],
    lengths: Sequence[Union[int, float]],
    generator: Optional[np.random.Generator] = None
) -> List[Subset[T]]:
    """
    Randomly split a dataset into non-overlapping new datasets.
    
    Args:
        dataset: Dataset to split
        lengths: Lengths or fractions of splits
        generator: Random number generator for reproducibility
    
    Returns:
        List of Subset objects
    
    Example:
        >>> dataset = TensorDataset(np.random.randn(100, 10))
        >>> train, val, test = random_split(dataset, [0.8, 0.1, 0.1])
    """
    if generator is None:
        generator = np.random.default_rng()
    
    total_length = len(dataset)
    
    # Handle fractional lengths
    if all(isinstance(l, float) for l in lengths):
        if abs(sum(lengths) - 1.0) > 1e-6:
            raise ValueError("Fractions must sum to 1")
        lengths = [int(l * total_length) for l in lengths]
        # Handle rounding errors
        diff = total_length - sum(lengths)
        lengths[0] += diff
    
    if sum(lengths) != total_length:
        raise ValueError(f"Sum of lengths ({sum(lengths)}) != dataset length ({total_length})")
    
    indices = generator.permutation(total_length).tolist()
    
    subsets = []
    offset = 0
    for length in lengths:
        subsets.append(Subset(dataset, indices[offset:offset + length]))
        offset += length
    
    return subsets


# Samplers

class Sampler(ABC, Generic[T_co]):
    """
    Base class for all samplers.
    
    Samplers define the order of samples yielded by the DataLoader.
    """
    
    @abstractmethod
    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError
    
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError


class SequentialSampler(Sampler[int]):
    """
    Sample elements sequentially in order.
    
    Args:
        data_source: Dataset to sample from
    """
    
    def __init__(self, data_source: Dataset):
        self.data_source = data_source
    
    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))
    
    def __len__(self) -> int:
        return len(self.data_source)


class RandomSampler(Sampler[int]):
    """
    Sample elements randomly without replacement.
    
    Args:
        data_source: Dataset to sample from
        replacement: If True, sample with replacement
        num_samples: Number of samples to draw. If None, use len(data_source)
        generator: Random number generator
    """
    
    def __init__(
        self,
        data_source: Dataset,
        replacement: bool = False,
        num_samples: Optional[int] = None,
        generator: Optional[np.random.Generator] = None
    ):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator if generator is not None else np.random.default_rng()
    
    @property
    def num_samples(self) -> int:
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples
    
    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.replacement:
            for _ in range(self.num_samples):
                yield int(self.generator.integers(n))
        else:
            yield from self.generator.permutation(n).tolist()
    
    def __len__(self) -> int:
        return self.num_samples


class SubsetRandomSampler(Sampler[int]):
    """
    Sample randomly from a given list of indices.
    
    Args:
        indices: Sequence of indices to sample from
        generator: Random number generator
    """
    
    def __init__(
        self,
        indices: Sequence[int],
        generator: Optional[np.random.Generator] = None
    ):
        self.indices = list(indices)
        self.generator = generator if generator is not None else np.random.default_rng()
    
    def __iter__(self) -> Iterator[int]:
        for i in self.generator.permutation(len(self.indices)):
            yield self.indices[i]
    
    def __len__(self) -> int:
        return len(self.indices)


class WeightedRandomSampler(Sampler[int]):
    """
    Sample elements with given probabilities (weights).
    
    Args:
        weights: Sequence of weights (not necessarily summing to 1)
        num_samples: Number of samples to draw
        replacement: If True, sample with replacement
        generator: Random number generator
    """
    
    def __init__(
        self,
        weights: Sequence[float],
        num_samples: int,
        replacement: bool = True,
        generator: Optional[np.random.Generator] = None
    ):
        if not replacement and num_samples > len(weights):
            raise ValueError(
                "num_samples cannot exceed number of weights when sampling without replacement"
            )
        
        self.weights = np.array(weights, dtype=np.float64)
        self.weights /= self.weights.sum()  # Normalize
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator if generator is not None else np.random.default_rng()
    
    def __iter__(self) -> Iterator[int]:
        indices = self.generator.choice(
            len(self.weights),
            size=self.num_samples,
            replace=self.replacement,
            p=self.weights
        )
        yield from indices.tolist()
    
    def __len__(self) -> int:
        return self.num_samples


class BatchSampler(Sampler[List[int]]):
    """
    Wrap a sampler to yield batches of indices.
    
    Args:
        sampler: Base sampler
        batch_size: Size of each batch
        drop_last: If True, drop the last incomplete batch
    """
    
    def __init__(
        self,
        sampler: Sampler[int],
        batch_size: int,
        drop_last: bool = False
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
    
    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch
    
    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class DistributedSampler(Sampler[int]):
    """
    Sampler that restricts data loading to a subset for distributed training.
    
    Args:
        dataset: Dataset to sample from
        num_replicas: Number of processes in distributed training
        rank: Rank of current process
        shuffle: Whether to shuffle indices
        seed: Random seed
        drop_last: Whether to drop tail data to make evenly divisible
    """
    
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False
    ):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        
        if drop_last and len(dataset) % num_replicas != 0:
            self.num_samples = len(dataset) // num_replicas
        else:
            self.num_samples = (len(dataset) + num_replicas - 1) // num_replicas
        
        self.total_size = self.num_samples * num_replicas
    
    def __iter__(self) -> Iterator[int]:
        generator = np.random.default_rng(self.seed + self.epoch)
        
        if self.shuffle:
            indices = generator.permutation(len(self.dataset)).tolist()
        else:
            indices = list(range(len(self.dataset)))
        
        if not self.drop_last:
            # Pad to make evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size > 0:
                indices += indices[:padding_size]
        else:
            # Remove tail data
            indices = indices[:self.total_size]
        
        assert len(indices) == self.total_size
        
        # Subsample for this rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        
        yield from indices
    
    def __len__(self) -> int:
        return self.num_samples
    
    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for shuffling."""
        self.epoch = epoch


# Collate Functions

def default_convert(data: Any) -> Any:
    """Convert data to numpy array if possible."""
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, (list, tuple)):
        return np.array(data)
    elif isinstance(data, (int, float)):
        return np.array(data)
    return data


def default_collate(batch: List[Any]) -> Any:
    """
    Default collate function that stacks samples into a batch.
    
    Handles:
        - Numpy arrays: Stack along new batch dimension
        - Tuples/Lists: Collate each element recursively
        - Dicts: Collate each value recursively
        - Scalars: Stack into array
    
    Args:
        batch: List of samples from dataset
    
    Returns:
        Collated batch
    """
    if len(batch) == 0:
        raise ValueError("Cannot collate empty batch")
    
    elem = batch[0]
    
    if isinstance(elem, np.ndarray):
        return np.stack(batch, axis=0)
    
    elif isinstance(elem, (int, float, np.integer, np.floating)):
        return np.array(batch)
    
    elif isinstance(elem, str):
        return batch
    
    elif isinstance(elem, tuple):
        if hasattr(elem, '_fields'):  # namedtuple
            return type(elem)(*(default_collate(samples) for samples in zip(*batch)))
        return tuple(default_collate(samples) for samples in zip(*batch))
    
    elif isinstance(elem, list):
        return [default_collate(samples) for samples in zip(*batch)]
    
    elif isinstance(elem, dict):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    
    else:
        return batch


# DataLoader

class _DataLoaderIter:
    """Iterator for DataLoader."""
    
    def __init__(self, loader: 'DataLoader'):
        self.loader = loader
        self.dataset = loader.dataset
        self.batch_size = loader.batch_size
        self.collate_fn = loader.collate_fn
        self.drop_last = loader.drop_last
        
        if loader.batch_sampler is not None:
            self.sampler_iter = iter(loader.batch_sampler)
            self.batch_sampler_mode = True
        else:
            self.sampler_iter = iter(loader.sampler)
            self.batch_sampler_mode = False
        
        self._batch = []
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.batch_sampler_mode:
            # BatchSampler provides list of indices
            indices = next(self.sampler_iter)
            batch = [self.dataset[i] for i in indices]
            return self.collate_fn(batch)
        else:
            # Collect batch_size samples
            while len(self._batch) < self.batch_size:
                try:
                    idx = next(self.sampler_iter)
                    self._batch.append(self.dataset[idx])
                except StopIteration:
                    if self._batch and not self.drop_last:
                        batch = self._batch
                        self._batch = []
                        return self.collate_fn(batch)
                    raise
            
            batch = self._batch[:self.batch_size]
            self._batch = self._batch[self.batch_size:]
            return self.collate_fn(batch)


class DataLoader(Generic[T_co]):
    """
    standard-style DataLoader for iterating over datasets.
    
    Combines a dataset and a sampler, providing batched iteration.
    
    Args:
        dataset: Dataset to load data from
        batch_size: Samples per batch (default: 1)
        shuffle: Shuffle data each epoch (default: False)
        sampler: Custom sampler (mutually exclusive with shuffle)
        batch_sampler: Custom batch sampler (overrides batch_size, shuffle, sampler, drop_last)
        num_workers: Not implemented (for API compatibility)
        collate_fn: Function to collate samples into batch
        pin_memory: Not implemented (for API compatibility)
        drop_last: Drop last incomplete batch
        timeout: Not implemented (for API compatibility)
        worker_init_fn: Not implemented (for API compatibility)
        generator: Random number generator for shuffling
        prefetch_factor: Not implemented (for API compatibility)
        persistent_workers: Not implemented (for API compatibility)
    
    Example:
        >>> dataset = TensorDataset(np.random.randn(100, 10), np.random.randint(0, 2, 100))
        >>> loader = DataLoader(dataset, batch_size=16, shuffle=True)
        >>> for X_batch, y_batch in loader:
        ...     # Training step
        ...     pass
    """
    
    def __init__(
        self,
        dataset: Dataset[T_co],
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: Optional[Sampler] = None,
        batch_sampler: Optional[BatchSampler] = None,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[Callable] = None,
        generator: Optional[np.random.Generator] = None,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        
        # Handle sampler
        if batch_sampler is not None:
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError(
                    "batch_sampler is mutually exclusive with batch_size, shuffle, sampler, and drop_last"
                )
            self.batch_sampler = batch_sampler
            self.sampler = batch_sampler.sampler
            self.batch_size = batch_sampler.batch_size
        else:
            self.batch_sampler = None
            if sampler is not None:
                if shuffle:
                    raise ValueError("sampler is mutually exclusive with shuffle")
                self.sampler = sampler
            elif shuffle:
                self.sampler = RandomSampler(dataset, generator=generator)
            else:
                self.sampler = SequentialSampler(dataset)
        
        # Collate function
        self.collate_fn = collate_fn if collate_fn is not None else default_collate
        
        self.generator = generator
    
    def __iter__(self) -> Iterator[T_co]:
        return _DataLoaderIter(self)
    
    def __len__(self) -> int:
        if self.batch_sampler is not None:
            return len(self.batch_sampler)
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size


# Utility Datasets

class ImageFolder(Dataset):
    """
    Dataset for loading images from a directory structure.
    
    Expected directory structure:
        root/class1/image1.jpg
        root/class1/image2.jpg
        root/class2/image3.jpg
        ...
    
    Args:
        root: Root directory path
        transform: Optional transform to apply to images
        target_transform: Optional transform to apply to targets
        extensions: Valid file extensions
        loader: Function to load an image from path
    """
    
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
    
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        extensions: Optional[Tuple[str, ...]] = None,
        loader: Optional[Callable] = None
    ):
        import os
        
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.extensions = extensions if extensions else self.IMG_EXTENSIONS
        
        if loader is None:
            self.loader = self._default_loader
        else:
            self.loader = loader
        
        # Find classes
        self.classes = sorted(
            entry for entry in os.listdir(root)
            if os.path.isdir(os.path.join(root, entry))
        )
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Find all images
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for filename in sorted(os.listdir(class_dir)):
                if self._has_valid_extension(filename):
                    path = os.path.join(class_dir, filename)
                    self.samples.append((path, class_idx))
        
        self.targets = [s[1] for s in self.samples]
    
    def _has_valid_extension(self, filename: str) -> bool:
        return filename.lower().endswith(self.extensions)
    
    @staticmethod
    def _default_loader(path: str) -> np.ndarray:
        """Load image using available backend."""
        try:
            from PIL import Image
            with open(path, 'rb') as f:
                img = Image.open(f)
                return np.array(img.convert('RGB'))
        except ImportError:
            pass
        
        try:
            import cv2
            img = cv2.imread(path)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except ImportError:
            pass
        
        raise RuntimeError(f"Cannot load image: {path}. Install PIL or cv2.")
    
    def __getitem__(self, index: int) -> Tuple[Any, int]:
        path, target = self.samples[index]
        sample = self.loader(path)
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return sample, target
    
    def __len__(self) -> int:
        return len(self.samples)
