# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Learning Rate Schedulers.

Neurova implementation of learning rate scheduling strategies
for optimizing neural network training.
"""

from __future__ import annotations
import math
import numpy as np
from typing import List, Optional, Callable, Union
from abc import ABC, abstractmethod


class _LRScheduler(ABC):
    """
    Base class for all learning rate schedulers.
    
    Neurova implementation for dynamic learning rate adjustment.
    
    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer
    last_epoch : int, default=-1
        The index of last epoch
    verbose : bool, default=False
        If True, prints a message to stdout for each update
    """
    
    def __init__(self, optimizer, last_epoch: int = -1, verbose: bool = False):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.last_epoch = last_epoch
        self.verbose = verbose
        self._last_lr = optimizer.lr
        
        if last_epoch == -1:
            self.step()
    
    @abstractmethod
    def get_lr(self) -> float:
        """Compute learning rate using chainable form of the scheduler."""
        raise NotImplementedError
    
    def step(self, epoch: Optional[int] = None) -> None:
        """
        Step the scheduler to update learning rate.
        
        Parameters
        ----------
        epoch : int, optional
            The epoch number. If None, uses last_epoch + 1
        """
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        
        lr = self.get_lr()
        self.optimizer.lr = lr
        self._last_lr = lr
        
        if self.verbose:
            print(f'Epoch {self.last_epoch}: adjusting learning rate to {lr:.6f}')
    
    def get_last_lr(self) -> float:
        """Return last computed learning rate."""
        return self._last_lr
    
    def state_dict(self) -> dict:
        """Return the state of the scheduler as a dict."""
        return {
            'last_epoch': self.last_epoch,
            'base_lr': self.base_lr,
            '_last_lr': self._last_lr,
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        """Load the scheduler state."""
        self.last_epoch = state_dict['last_epoch']
        self.base_lr = state_dict['base_lr']
        self._last_lr = state_dict['_last_lr']


class StepLR(_LRScheduler):
    """
    Decays learning rate by gamma every step_size epochs.
    
    Neurova implementation.
    
    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer
    step_size : int
        Period of learning rate decay
    gamma : float, default=0.1
        Multiplicative factor of learning rate decay
    last_epoch : int, default=-1
        The index of last epoch
    verbose : bool, default=False
        If True, prints update message
    
    Examples
    --------
    >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    >>> for epoch in range(100):
    ...     train(...)
    ...     validate(...)
    ...     scheduler.step()
    """
    
    def __init__(
        self,
        optimizer,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> float:
        """Compute decayed learning rate."""
        return self.base_lr * (self.gamma ** (self.last_epoch // self.step_size))


class MultiStepLR(_LRScheduler):
    """
    Decays learning rate by gamma at each milestone.
    
    Neurova implementation.
    
    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer
    milestones : list of int
        List of epoch indices to decay LR
    gamma : float, default=0.1
        Multiplicative factor of learning rate decay
    last_epoch : int, default=-1
        The index of last epoch
    verbose : bool, default=False
        If True, prints update message
    
    Examples
    --------
    >>> scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
    """
    
    def __init__(
        self,
        optimizer,
        milestones: List[int],
        gamma: float = 0.1,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.milestones = sorted(milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> float:
        """Compute decayed learning rate."""
        count = sum(1 for m in self.milestones if self.last_epoch >= m)
        return self.base_lr * (self.gamma ** count)


class ExponentialLR(_LRScheduler):
    """
    Decays learning rate by gamma every epoch.
    
    Neurova implementation.
    
    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer
    gamma : float
        Multiplicative factor of learning rate decay
    last_epoch : int, default=-1
        The index of last epoch
    verbose : bool, default=False
        If True, prints update message
    
    Examples
    --------
    >>> scheduler = ExponentialLR(optimizer, gamma=0.95)
    """
    
    def __init__(
        self,
        optimizer,
        gamma: float,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> float:
        """Compute exponentially decayed learning rate."""
        return self.base_lr * (self.gamma ** self.last_epoch)


class CosineAnnealingLR(_LRScheduler):
    """
    Cosine annealing learning rate schedule.
    
    Neurova implementation.
    
    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer
    T_max : int
        Maximum number of iterations
    eta_min : float, default=0
        Minimum learning rate
    last_epoch : int, default=-1
        The index of last epoch
    verbose : bool, default=False
        If True, prints update message
    
    Examples
    --------
    >>> scheduler = CosineAnnealingLR(optimizer, T_max=100)
    """
    
    def __init__(
        self,
        optimizer,
        T_max: int,
        eta_min: float = 0,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> float:
        """Compute cosine annealed learning rate."""
        return self.eta_min + (self.base_lr - self.eta_min) * (
            1 + math.cos(math.pi * self.last_epoch / self.T_max)
        ) / 2


class CosineAnnealingWarmRestarts(_LRScheduler):
    """
    Cosine annealing with warm restarts.
    
    Neurova implementation.
    
    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer
    T_0 : int
        Number of iterations for the first restart
    T_mult : int, default=1
        Factor to increase T_i after a restart
    eta_min : float, default=0
        Minimum learning rate
    last_epoch : int, default=-1
        The index of last epoch
    verbose : bool, default=False
        If True, prints update message
    
    Examples
    --------
    >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    """
    
    def __init__(
        self,
        optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_i = T_0
        self.T_cur = 0
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> float:
        """Compute learning rate with warm restarts."""
        return self.eta_min + (self.base_lr - self.eta_min) * (
            1 + math.cos(math.pi * self.T_cur / self.T_i)
        ) / 2
    
    def step(self, epoch: Optional[int] = None) -> None:
        """Step with restart logic."""
        if epoch is None:
            self.T_cur += 1
            if self.T_cur >= self.T_i:
                self.T_cur = 0
                self.T_i = self.T_i * self.T_mult
        else:
            self.T_cur = epoch % self.T_i
        
        super().step(epoch)


class CyclicLR(_LRScheduler):
    """
    Cyclic learning rate policy.
    
    Neurova implementation.
    
    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer
    base_lr : float
        Initial learning rate (lower boundary)
    max_lr : float
        Upper learning rate boundary
    step_size_up : int, default=2000
        Number of training iterations in the increasing half
    step_size_down : int, optional
        Number of training iterations in the decreasing half
    mode : str, default='triangular'
        One of 'triangular', 'triangular2', 'exp_range'
    gamma : float, default=1.0
        Constant for 'exp_range' mode scaling
    scale_fn : callable, optional
        Custom scaling function
    cycle_momentum : bool, default=True
        If True, momentum is cycled inversely to LR
    last_epoch : int, default=-1
        The index of last epoch
    verbose : bool, default=False
        If True, prints update message
    
    Examples
    --------
    >>> scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.1)
    """
    
    def __init__(
        self,
        optimizer,
        base_lr: float,
        max_lr: float,
        step_size_up: int = 2000,
        step_size_down: Optional[int] = None,
        mode: str = 'triangular',
        gamma: float = 1.0,
        scale_fn: Optional[Callable[[float], float]] = None,
        cycle_momentum: bool = True,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.base_lr_val = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down or step_size_up
        self.mode = mode
        self.gamma = gamma
        self.cycle_momentum = cycle_momentum
        
        if scale_fn is not None:
            self.scale_fn = scale_fn
        elif mode == 'triangular':
            self.scale_fn = lambda x: 1.0
        elif mode == 'triangular2':
            self.scale_fn = lambda x: 1 / (2.0 ** (x - 1))
        elif mode == 'exp_range':
            self.scale_fn = lambda x: gamma ** (x)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        super().__init__(optimizer, last_epoch, verbose)
        self.base_lr = self.base_lr_val
    
    def get_lr(self) -> float:
        """Compute cyclic learning rate."""
        cycle = math.floor(1 + self.last_epoch / (self.step_size_up + self.step_size_down))
        x = abs(self.last_epoch / self.step_size_up - 2 * cycle + 1)
        scale = self.scale_fn(cycle)
        lr = self.base_lr_val + (self.max_lr - self.base_lr_val) * max(0, (1 - x)) * scale
        return lr


class OneCycleLR(_LRScheduler):
    """
    One cycle learning rate policy.
    
    Neurova implementation.
    
    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer
    max_lr : float
        Upper learning rate boundary
    total_steps : int
        Total number of training steps
    pct_start : float, default=0.3
        Percentage of cycle spent increasing LR
    anneal_strategy : str, default='cos'
        Annealing strategy: 'cos' or 'linear'
    div_factor : float, default=25.0
        Initial LR = max_lr / div_factor
    final_div_factor : float, default=1e4
        Final LR = initial_lr / final_div_factor
    last_epoch : int, default=-1
        The index of last epoch
    verbose : bool, default=False
        If True, prints update message
    
    Examples
    --------
    >>> scheduler = OneCycleLR(optimizer, max_lr=0.1, total_steps=1000)
    """
    
    def __init__(
        self,
        optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        anneal_strategy: str = 'cos',
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        self.initial_lr = max_lr / div_factor
        self.final_lr = self.initial_lr / final_div_factor
        self.step_up = int(total_steps * pct_start)
        self.step_down = total_steps - self.step_up
        
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> float:
        """Compute one-cycle learning rate."""
        if self.last_epoch < self.step_up:
            # warmup phase
            pct = self.last_epoch / self.step_up
            return self.initial_lr + (self.max_lr - self.initial_lr) * pct
        else:
            # annealing phase
            pct = (self.last_epoch - self.step_up) / self.step_down
            if self.anneal_strategy == 'cos':
                return self.final_lr + (self.max_lr - self.final_lr) * (
                    1 + math.cos(math.pi * pct)
                ) / 2
            else:
                return self.max_lr - (self.max_lr - self.final_lr) * pct


class ReduceLROnPlateau:
    """
    Reduce learning rate when a metric has stopped improving.
    
    Neurova implementation.
    
    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer
    mode : str, default='min'
        One of 'min' or 'max'
    factor : float, default=0.1
        Factor by which the learning rate will be reduced
    patience : int, default=10
        Number of epochs with no improvement before LR reduction
    threshold : float, default=1e-4
        Threshold for measuring improvement
    threshold_mode : str, default='rel'
        One of 'rel' or 'abs'
    cooldown : int, default=0
        Number of epochs to wait before resuming normal operation
    min_lr : float, default=0
        Lower bound on the learning rate
    eps : float, default=1e-8
        Minimal decay applied to lr
    verbose : bool, default=False
        If True, prints update message
    
    Examples
    --------
    >>> scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10)
    >>> for epoch in range(100):
    ...     train(...)
    ...     val_loss = validate(...)
    ...     scheduler.step(val_loss)
    """
    
    def __init__(
        self,
        optimizer,
        mode: str = 'min',
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = 'rel',
        cooldown: int = 0,
        min_lr: float = 0,
        eps: float = 1e-8,
        verbose: bool = False
    ):
        if mode not in ('min', 'max'):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
        if factor >= 1.0:
            raise ValueError(f"factor should be < 1.0, got {factor}")
        
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.eps = eps
        self.verbose = verbose
        
        self.cooldown_counter = 0
        self.best = None
        self.num_bad_epochs = 0
        self._last_lr = optimizer.lr
        
        self._init_is_better(mode)
    
    def _init_is_better(self, mode: str) -> None:
        if mode == 'min':
            self.best = float('inf')
        else:
            self.best = float('-inf')
    
    def _is_better(self, current: float) -> bool:
        if self.mode == 'min':
            if self.threshold_mode == 'rel':
                return current < self.best * (1 - self.threshold)
            else:
                return current < self.best - self.threshold
        else:
            if self.threshold_mode == 'rel':
                return current > self.best * (1 + self.threshold)
            else:
                return current > self.best + self.threshold
    
    def step(self, metric: float) -> None:
        """
        Update learning rate based on metric.
        
        Parameters
        ----------
        metric : float
            The metric value to check for improvement
        """
        current = metric
        
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0
        
        if self._is_better(current):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        if self.num_bad_epochs > self.patience:
            self._reduce_lr()
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
    
    def _reduce_lr(self) -> None:
        """Reduce learning rate."""
        old_lr = self.optimizer.lr
        new_lr = max(old_lr * self.factor, self.min_lr)
        
        if old_lr - new_lr > self.eps:
            self.optimizer.lr = new_lr
            self._last_lr = new_lr
            if self.verbose:
                print(f'Reducing learning rate to {new_lr:.6f}')
    
    def get_last_lr(self) -> float:
        """Return last computed learning rate."""
        return self._last_lr


class LinearLR(_LRScheduler):
    """
    Linear warmup/decay learning rate.
    
    Neurova implementation.
    
    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer
    start_factor : float, default=1.0/3
        Start factor for the warmup
    end_factor : float, default=1.0
        End factor for the warmup
    total_iters : int, default=5
        Number of iterations for the warmup
    last_epoch : int, default=-1
        The index of last epoch
    verbose : bool, default=False
        If True, prints update message
    
    Examples
    --------
    >>> scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=10)
    """
    
    def __init__(
        self,
        optimizer,
        start_factor: float = 1.0 / 3,
        end_factor: float = 1.0,
        total_iters: int = 5,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> float:
        """Compute linearly interpolated learning rate."""
        if self.last_epoch >= self.total_iters:
            return self.base_lr * self.end_factor
        
        pct = self.last_epoch / self.total_iters
        factor = self.start_factor + (self.end_factor - self.start_factor) * pct
        return self.base_lr * factor


class PolynomialLR(_LRScheduler):
    """
    Polynomial decay learning rate.
    
    Neurova implementation.
    
    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer
    total_iters : int
        Total number of training steps
    power : float, default=1.0
        Power of polynomial decay
    last_epoch : int, default=-1
        The index of last epoch
    verbose : bool, default=False
        If True, prints update message
    
    Examples
    --------
    >>> scheduler = PolynomialLR(optimizer, total_iters=100, power=2.0)
    """
    
    def __init__(
        self,
        optimizer,
        total_iters: int,
        power: float = 1.0,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.total_iters = total_iters
        self.power = power
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> float:
        """Compute polynomial decayed learning rate."""
        if self.last_epoch >= self.total_iters:
            return 0.0
        
        decay = (1 - self.last_epoch / self.total_iters) ** self.power
        return self.base_lr * decay


class ConstantLR(_LRScheduler):
    """
    Constant factor for warmup.
    
    Neurova implementation.
    
    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer
    factor : float, default=1.0/3
        Factor to multiply LR during warmup
    total_iters : int, default=5
        Duration of warmup
    last_epoch : int, default=-1
        The index of last epoch
    verbose : bool, default=False
        If True, prints update message
    
    Examples
    --------
    >>> scheduler = ConstantLR(optimizer, factor=0.1, total_iters=10)
    """
    
    def __init__(
        self,
        optimizer,
        factor: float = 1.0 / 3,
        total_iters: int = 5,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.factor = factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> float:
        """Compute constant learning rate."""
        if self.last_epoch < self.total_iters:
            return self.base_lr * self.factor
        return self.base_lr


class SequentialLR:
    """
    Chain multiple schedulers sequentially.
    
    Neurova implementation.
    
    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer
    schedulers : list
        List of schedulers to chain
    milestones : list of int
        Epochs at which to switch to next scheduler
    last_epoch : int, default=-1
        The index of last epoch
    verbose : bool, default=False
        If True, prints update message
    
    Examples
    --------
    >>> s1 = LinearLR(optimizer, start_factor=0.1, total_iters=10)
    >>> s2 = CosineAnnealingLR(optimizer, T_max=100)
    >>> scheduler = SequentialLR(optimizer, [s1, s2], milestones=[10])
    """
    
    def __init__(
        self,
        optimizer,
        schedulers: list,
        milestones: List[int],
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.optimizer = optimizer
        self.schedulers = schedulers
        self.milestones = milestones
        self.last_epoch = last_epoch
        self.verbose = verbose
        self._last_lr = optimizer.lr
    
    def step(self) -> None:
        """Step the appropriate scheduler."""
        self.last_epoch += 1
        
        # find current scheduler index
        idx = 0
        for i, m in enumerate(self.milestones):
            if self.last_epoch >= m:
                idx = i + 1
        
        if idx < len(self.schedulers):
            self.schedulers[idx].step()
            self._last_lr = self.schedulers[idx].get_last_lr()
    
    def get_last_lr(self) -> float:
        """Return last computed learning rate."""
        return self._last_lr


class ChainedScheduler:
    """
    Chain schedulers together (multiplicative).
    
    Neurova implementation.
    
    Parameters
    ----------
    schedulers : list
        List of schedulers to chain
    
    Examples
    --------
    >>> s1 = LinearLR(optimizer, start_factor=0.1, total_iters=10)
    >>> s2 = ExponentialLR(optimizer, gamma=0.95)
    >>> scheduler = ChainedScheduler([s1, s2])
    """
    
    def __init__(self, schedulers: list):
        self.schedulers = schedulers
        self._last_lr = schedulers[0].optimizer.lr if schedulers else 0
    
    def step(self) -> None:
        """Step all schedulers."""
        for scheduler in self.schedulers:
            scheduler.step()
        
        if self.schedulers:
            self._last_lr = self.schedulers[-1].get_last_lr()
    
    def get_last_lr(self) -> float:
        """Return last computed learning rate."""
        return self._last_lr


class LambdaLR(_LRScheduler):
    """
    Custom lambda function scheduler.
    
    Neurova implementation.
    
    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer
    lr_lambda : callable
        Function that computes multiplicative factor given epoch
    last_epoch : int, default=-1
        The index of last epoch
    verbose : bool, default=False
        If True, prints update message
    
    Examples
    --------
    >>> scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
    """
    
    def __init__(
        self,
        optimizer,
        lr_lambda: Callable[[int], float],
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.lr_lambda = lr_lambda
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> float:
        """Compute learning rate using lambda function."""
        return self.base_lr * self.lr_lambda(self.last_epoch)


class MultiplicativeLR(_LRScheduler):
    """
    Multiply LR by lambda each epoch.
    
    Neurova implementation.
    
    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer
    lr_lambda : callable
        Function that computes multiplicative factor given epoch
    last_epoch : int, default=-1
        The index of last epoch
    verbose : bool, default=False
        If True, prints update message
    
    Examples
    --------
    >>> scheduler = MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.95)
    """
    
    def __init__(
        self,
        optimizer,
        lr_lambda: Callable[[int], float],
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.lr_lambda = lr_lambda
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> float:
        """Compute multiplicative learning rate."""
        if self.last_epoch == 0:
            return self.base_lr
        return self._last_lr * self.lr_lambda(self.last_epoch)


# Export all schedulers
__all__ = [
    '_LRScheduler',
    'StepLR',
    'MultiStepLR',
    'ExponentialLR',
    'CosineAnnealingLR',
    'CosineAnnealingWarmRestarts',
    'CyclicLR',
    'OneCycleLR',
    'ReduceLROnPlateau',
    'LinearLR',
    'PolynomialLR',
    'ConstantLR',
    'SequentialLR',
    'ChainedScheduler',
    'LambdaLR',
    'MultiplicativeLR',
]
