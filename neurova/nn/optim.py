# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
Optimization algorithms.

Neurova implementation with SGD, Adam, RMSprop, etc.
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional
from neurova.nn.autograd import Parameter


class Optimizer:
    """
    Base class for all optimizers.
    
    Neurova implementation
    
    Parameters
    ----------
    params : list of Parameter
        Parameters to optimize
    lr : float
        Learning rate
    """
    
    def __init__(self, params: List[Parameter], lr: float = 0.01):
        self.params = list(params)
        self.lr = lr
        self.t = 0  # Time step for adaptive methods
    
    def step(self) -> None:
        """
        Perform a single optimization step.
        
        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses
        """
        raise NotImplementedError
    
    def zero_grad(self) -> None:
        """Zero out all parameter gradients."""
        for param in self.params:
            param.zero_grad()


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    
    Neurova implementation
    
    Parameters
    ----------
    params : list of Parameter
        Parameters to optimize
    lr : float, default=0.01
        Learning rate
    momentum : float, default=0
        Momentum factor
    weight_decay : float, default=0
        Weight decay (L2 penalty)
    nesterov : bool, default=False
        Use Nesterov momentum
    
    Examples
    --------
    >>> optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    >>> optimizer.zero_grad()
    >>> loss.backward()
    >>> optimizer.step()
    """
    
    def __init__(
        self,
        params: List[Parameter],
        lr: float = 0.01,
        momentum: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False
    ):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        
        # velocity for momentum
        self.velocity = [np.zeros_like(p.data) for p in self.params]
    
    def step(self) -> None:
        """Perform SGD update."""
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            grad = param.grad.copy()
            
            # weight decay
            if self.weight_decay != 0:
                grad += self.weight_decay * param.data
            
            # momentum
            if self.momentum != 0:
                self.velocity[i] = self.momentum * self.velocity[i] + grad
                
                if self.nesterov:
                    grad = grad + self.momentum * self.velocity[i]
                else:
                    grad = self.velocity[i]
            
            # update parameters
            param.data -= self.lr * grad


class Adam(Optimizer):
    """
    Adam optimizer (Adaptive Moment Estimation).
    
    Neurova implementation
    
    Parameters
    ----------
    params : list of Parameter
        Parameters to optimize
    lr : float, default=0.001
        Learning rate
    betas : tuple of float, default=(0.9, 0.999)
        Coefficients for computing running averages
    eps : float, default=1e-8
        Term for numerical stability
    weight_decay : float, default=0
        Weight decay (L2 penalty)
    
    Examples
    --------
    >>> optimizer = Adam(model.parameters(), lr=0.001)
    >>> optimizer.zero_grad()
    >>> loss.backward()
    >>> optimizer.step()
    """
    
    def __init__(
        self,
        params: List[Parameter],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0
    ):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # first and second moment estimates
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
    
    def step(self) -> None:
        """Perform Adam update."""
        self.t += 1
        
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            grad = param.grad.copy()
            
            # weight decay
            if self.weight_decay != 0:
                grad += self.weight_decay * param.data
            
            # update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # update parameters
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Adam):
    """
    AdamW optimizer (Adam with decoupled weight decay).
    
    Neurova implementation
    
    Parameters
    ----------
    params : list of Parameter
        Parameters to optimize
    lr : float, default=0.001
        Learning rate
    betas : tuple of float, default=(0.9, 0.999)
        Coefficients for computing running averages
    eps : float, default=1e-8
        Term for numerical stability
    weight_decay : float, default=0.01
        Weight decay coefficient
    """
    
    def __init__(
        self,
        params: List[Parameter],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ):
        super().__init__(params, lr, betas, eps, weight_decay=0)
        self.wd = weight_decay
    
    def step(self) -> None:
        """Perform AdamW update."""
        self.t += 1
        
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            grad = param.grad.copy()
            
            # update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # update parameters with decoupled weight decay
            param.data -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + 
                                    self.wd * param.data)


class RMSprop(Optimizer):
    """
    RMSprop optimizer.
    
    Neurova implementation
    
    Parameters
    ----------
    params : list of Parameter
        Parameters to optimize
    lr : float, default=0.01
        Learning rate
    alpha : float, default=0.99
        Smoothing constant
    eps : float, default=1e-8
        Term for numerical stability
    weight_decay : float, default=0
        Weight decay (L2 penalty)
    momentum : float, default=0
        Momentum factor
    """
    
    def __init__(
        self,
        params: List[Parameter],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0,
        momentum: float = 0
    ):
        super().__init__(params, lr)
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        
        # running average of squared gradients
        self.square_avg = [np.zeros_like(p.data) for p in self.params]
        
        if momentum > 0:
            self.momentum_buffer = [np.zeros_like(p.data) for p in self.params]
    
    def step(self) -> None:
        """Perform RMSprop update."""
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            grad = param.grad.copy()
            
            # weight decay
            if self.weight_decay != 0:
                grad += self.weight_decay * param.data
            
            # update running average of squared gradients
            self.square_avg[i] = (self.alpha * self.square_avg[i] + 
                                 (1 - self.alpha) * grad ** 2)
            
            # compute update
            avg = np.sqrt(self.square_avg[i]) + self.eps
            
            if self.momentum > 0:
                self.momentum_buffer[i] = (self.momentum * self.momentum_buffer[i] + 
                                          grad / avg)
                param.data -= self.lr * self.momentum_buffer[i]
            else:
                param.data -= self.lr * grad / avg


class Adagrad(Optimizer):
    """
    Adagrad optimizer.
    
    Neurova implementation
    
    Parameters
    ----------
    params : list of Parameter
        Parameters to optimize
    lr : float, default=0.01
        Learning rate
    eps : float, default=1e-10
        Term for numerical stability
    weight_decay : float, default=0
        Weight decay (L2 penalty)
    """
    
    def __init__(
        self,
        params: List[Parameter],
        lr: float = 0.01,
        eps: float = 1e-10,
        weight_decay: float = 0
    ):
        super().__init__(params, lr)
        self.eps = eps
        self.weight_decay = weight_decay
        
        # accumulated squared gradients
        self.sum_squares = [np.zeros_like(p.data) for p in self.params]
    
    def step(self) -> None:
        """Perform Adagrad update."""
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            grad = param.grad.copy()
            
            # weight decay
            if self.weight_decay != 0:
                grad += self.weight_decay * param.data
            
            # accumulate squared gradients
            self.sum_squares[i] += grad ** 2
            
            # update parameters
            param.data -= self.lr * grad / (np.sqrt(self.sum_squares[i]) + self.eps)


class Adadelta(Optimizer):
    """
    Adadelta optimizer.
    
    Neurova implementation
    
    Parameters
    ----------
    params : list of Parameter
        Parameters to optimize
    lr : float, default=1.0
        Learning rate (usually kept at 1.0)
    rho : float, default=0.9
        Coefficient for computing running averages
    eps : float, default=1e-6
        Term for numerical stability
    weight_decay : float, default=0
        Weight decay (L2 penalty)
    """
    
    def __init__(
        self,
        params: List[Parameter],
        lr: float = 1.0,
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0
    ):
        super().__init__(params, lr)
        self.rho = rho
        self.eps = eps
        self.weight_decay = weight_decay
        
        # running averages
        self.square_avg = [np.zeros_like(p.data) for p in self.params]
        self.acc_delta = [np.zeros_like(p.data) for p in self.params]
    
    def step(self) -> None:
        """Perform Adadelta update."""
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            grad = param.grad.copy()
            
            # weight decay
            if self.weight_decay != 0:
                grad += self.weight_decay * param.data
            
            # accumulate gradient
            self.square_avg[i] = (self.rho * self.square_avg[i] + 
                                 (1 - self.rho) * grad ** 2)
            
            # compute update
            std = np.sqrt(self.square_avg[i] + self.eps)
            delta = np.sqrt(self.acc_delta[i] + self.eps) / std * grad
            
            # accumulate updates
            self.acc_delta[i] = self.rho * self.acc_delta[i] + (1 - self.rho) * delta ** 2
            
            # update parameters
            param.data -= self.lr * delta


class Adamax(Optimizer):
    """
    Adamax optimizer (variant of Adam based on infinity norm).
    
    Neurova implementation
    """
    
    def __init__(
        self,
        params: List[Parameter],
        lr: float = 0.002,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0
    ):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.u = [np.zeros_like(p.data) for p in self.params]
    
    def step(self) -> None:
        """Perform Adamax update."""
        self.t += 1
        
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            grad = param.grad.copy()
            
            if self.weight_decay != 0:
                grad += self.weight_decay * param.data
            
            # update first moment
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # update infinity norm
            self.u[i] = np.maximum(self.beta2 * self.u[i], np.abs(grad))
            
            # bias correction
            bias_correction = 1 - self.beta1 ** self.t
            
            # update parameters
            param.data -= self.lr / bias_correction * self.m[i] / (self.u[i] + self.eps)


class NAdam(Optimizer):
    """
    NAdam optimizer (Adam with Nesterov momentum).
    
    Neurova implementation
    """
    
    def __init__(
        self,
        params: List[Parameter],
        lr: float = 0.002,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0
    ):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
    
    def step(self) -> None:
        """Perform NAdam update."""
        self.t += 1
        
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            grad = param.grad.copy()
            
            if self.weight_decay != 0:
                grad += self.weight_decay * param.data
            
            # update moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # nesterov momentum
            m_bar = self.beta1 * m_hat + (1 - self.beta1) / (1 - self.beta1 ** self.t) * grad
            
            # update parameters
            param.data -= self.lr * m_bar / (np.sqrt(v_hat) + self.eps)


class RAdam(Optimizer):
    """
    RAdam optimizer (Rectified Adam).
    
    Neurova implementation
    """
    
    def __init__(
        self,
        params: List[Parameter],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0
    ):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
    
    def step(self) -> None:
        """Perform RAdam update."""
        self.t += 1
        
        # compute rectification term
        rho_inf = 2.0 / (1 - self.beta2) - 1
        rho_t = rho_inf - 2 * self.t * (self.beta2 ** self.t) / (1 - self.beta2 ** self.t)
        
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            grad = param.grad.copy()
            
            if self.weight_decay != 0:
                grad += self.weight_decay * param.data
            
            # update moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            
            # variance rectification
            if rho_t > 4:
                v_hat = np.sqrt(self.v[i] / (1 - self.beta2 ** self.t))
                r_t = np.sqrt(((rho_t - 4) * (rho_t - 2) * rho_inf) / 
                             ((rho_inf - 4) * (rho_inf - 2) * rho_t))
                param.data -= self.lr * r_t * m_hat / (v_hat + self.eps)
            else:
                param.data -= self.lr * m_hat


class ASGD(Optimizer):
    """
    Averaged Stochastic Gradient Descent optimizer.
    
    Averages parameter updates for better generalization.
    
    Parameters
    ----------
    params : list of Parameter
        Parameters to optimize
    lr : float, default=0.01
        Learning rate
    lambd : float, default=1e-4
        Decay term
    alpha : float, default=0.75
        Power for eta update
    t0 : float, default=1e6
        Point at which to start averaging
    weight_decay : float, default=0
        Weight decay (L2 penalty)
    """
    
    def __init__(self, params: List[Parameter], lr: float = 0.01,
                 lambd: float = 1e-4, alpha: float = 0.75,
                 t0: float = 1e6, weight_decay: float = 0):
        super().__init__(params, lr)
        self.lambd = lambd
        self.alpha = alpha
        self.t0 = t0
        self.weight_decay = weight_decay
        
        # Averaged parameters
        self.ax = [np.zeros_like(p.data) for p in self.params]
        self.mu = 1.0
        self.eta = lr
    
    def step(self) -> None:
        """Perform ASGD update."""
        self.t += 1
        
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            grad = param.grad.copy()
            
            if self.weight_decay != 0:
                grad += self.weight_decay * param.data
            
            # Update parameters
            param.data -= self.eta * grad
            
            # Update averaged parameters
            if self.t >= self.t0:
                self.ax[i] = self.ax[i] + self.mu * (param.data - self.ax[i])
        
        # Update eta and mu
        self.eta = self.lr / ((1 + self.lambd * self.lr * self.t) ** self.alpha)
        if self.t >= self.t0:
            self.mu = 1.0 / max(1, self.t - self.t0 + 1)
    
    def get_averaged_params(self) -> List[np.ndarray]:
        """Get the averaged parameters."""
        return [ax.copy() for ax in self.ax]


class Rprop(Optimizer):
    """
    Resilient Backpropagation optimizer.
    
    Uses only the sign of gradients for updates.
    
    Parameters
    ----------
    params : list of Parameter
        Parameters to optimize
    lr : float, default=0.01
        Initial learning rate (step size)
    etas : tuple, default=(0.5, 1.2)
        Multiplicative factors (eta_minus, eta_plus)
    step_sizes : tuple, default=(1e-6, 50)
        Min and max step sizes
    """
    
    def __init__(self, params: List[Parameter], lr: float = 0.01,
                 etas: tuple = (0.5, 1.2), step_sizes: tuple = (1e-6, 50)):
        super().__init__(params, lr)
        self.eta_minus, self.eta_plus = etas
        self.step_min, self.step_max = step_sizes
        
        # Per-parameter step sizes and previous gradients
        self.step_sizes = [np.full_like(p.data, lr) for p in self.params]
        self.prev_grad = [np.zeros_like(p.data) for p in self.params]
    
    def step(self) -> None:
        """Perform Rprop update."""
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            grad = param.grad.copy()
            prev = self.prev_grad[i]
            step = self.step_sizes[i]
            
            # Compute sign changes
            sign_change = grad * prev
            
            # Where sign is same (positive), increase step
            pos_mask = sign_change > 0
            step[pos_mask] = np.minimum(step[pos_mask] * self.eta_plus, self.step_max)
            
            # Where sign changed (negative), decrease step
            neg_mask = sign_change < 0
            step[neg_mask] = np.maximum(step[neg_mask] * self.eta_minus, self.step_min)
            grad[neg_mask] = 0  # Don't update where sign changed
            
            # Update parameters
            param.data -= np.sign(grad) * step
            
            # Store gradient for next step
            self.prev_grad[i] = grad.copy()


class LBFGS(Optimizer):
    """
    Limited-memory BFGS optimizer.
    
    Quasi-Newton method for optimization. Requires closure for function evaluation.
    
    Parameters
    ----------
    params : list of Parameter
        Parameters to optimize
    lr : float, default=1.0
        Learning rate (step size multiplier)
    max_iter : int, default=20
        Maximum iterations per optimization step
    max_eval : int, default=25
        Maximum function evaluations per optimization step
    tolerance_grad : float, default=1e-7
        Termination tolerance on gradient norm
    tolerance_change : float, default=1e-9
        Termination tolerance on function value change
    history_size : int, default=100
        Size of history for L-BFGS
    line_search_fn : str, default='strong_wolfe'
        Line search function ('strong_wolfe' or None)
    """
    
    def __init__(self, params: List[Parameter], lr: float = 1.0,
                 max_iter: int = 20, max_eval: int = 25,
                 tolerance_grad: float = 1e-7, tolerance_change: float = 1e-9,
                 history_size: int = 100, line_search_fn: str = 'strong_wolfe'):
        super().__init__(params, lr)
        self.max_iter = max_iter
        self.max_eval = max_eval if max_eval is not None else int(max_iter * 1.25)
        self.tolerance_grad = tolerance_grad
        self.tolerance_change = tolerance_change
        self.history_size = history_size
        self.line_search_fn = line_search_fn
        
        # State
        self._n_iter = 0
        self._prev_loss = None
        self._prev_flat_grad = None
        
        # L-BFGS history
        self._s_history = []  # Parameter differences
        self._y_history = []  # Gradient differences
        self._rho_history = []  # 1 / (y.T @ s)
    
    def _gather_flat_grad(self) -> np.ndarray:
        """Gather gradients into a flat vector."""
        grads = []
        for param in self.params:
            if param.grad is not None:
                grads.append(param.grad.ravel())
            else:
                grads.append(np.zeros(param.data.size))
        return np.concatenate(grads)
    
    def _gather_flat_params(self) -> np.ndarray:
        """Gather parameters into a flat vector."""
        return np.concatenate([p.data.ravel() for p in self.params])
    
    def _set_flat_params(self, flat_params: np.ndarray) -> None:
        """Set parameters from a flat vector."""
        offset = 0
        for param in self.params:
            numel = param.data.size
            param.data = flat_params[offset:offset + numel].reshape(param.data.shape)
            offset += numel
    
    def _two_loop_recursion(self, flat_grad: np.ndarray) -> np.ndarray:
        """Compute search direction using L-BFGS two-loop recursion."""
        q = flat_grad.copy()
        history_len = len(self._s_history)
        
        if history_len == 0:
            return q
        
        alphas = []
        
        # First loop (reverse order)
        for i in range(history_len - 1, -1, -1):
            alpha = self._rho_history[i] * np.dot(self._s_history[i], q)
            alphas.insert(0, alpha)
            q = q - alpha * self._y_history[i]
        
        # Initial Hessian approximation
        s = self._s_history[-1]
        y = self._y_history[-1]
        gamma = np.dot(s, y) / (np.dot(y, y) + 1e-10)
        r = gamma * q
        
        # Second loop (forward order)
        for i in range(history_len):
            beta = self._rho_history[i] * np.dot(self._y_history[i], r)
            r = r + self._s_history[i] * (alphas[i] - beta)
        
        return r
    
    def step(self, closure=None):
        """
        Perform a single L-BFGS optimization step.
        
        Parameters
        ----------
        closure : callable
            A closure that re-evaluates the model and returns the loss.
            Must call backward() before returning.
        """
        if closure is None:
            raise RuntimeError("LBFGS requires a closure that returns the loss")
        
        # Initial evaluation
        loss = float(closure())
        flat_grad = self._gather_flat_grad()
        
        # Check for convergence
        grad_norm = np.linalg.norm(flat_grad)
        if grad_norm < self.tolerance_grad:
            return loss
        
        # Compute search direction
        d = -self._two_loop_recursion(flat_grad)
        
        # Line search
        current_params = self._gather_flat_params()
        step_size = self.lr
        
        if self.line_search_fn == 'strong_wolfe':
            step_size = self._strong_wolfe_line_search(closure, current_params, d, flat_grad, loss)
        
        # Update parameters
        new_params = current_params + step_size * d
        self._set_flat_params(new_params)
        
        # Re-evaluate for history update
        new_loss = float(closure())
        new_flat_grad = self._gather_flat_grad()
        
        # Update history
        s = step_size * d
        y = new_flat_grad - flat_grad
        
        ys = np.dot(y, s)
        if ys > 1e-10:  # Curvature condition
            if len(self._s_history) >= self.history_size:
                self._s_history.pop(0)
                self._y_history.pop(0)
                self._rho_history.pop(0)
            
            self._s_history.append(s)
            self._y_history.append(y)
            self._rho_history.append(1.0 / ys)
        
        self._n_iter += 1
        return new_loss
    
    def _strong_wolfe_line_search(self, closure, x0, d, g0, f0,
                                   c1: float = 1e-4, c2: float = 0.9,
                                   max_iter: int = 25) -> float:
        """Strong Wolfe line search."""
        alpha = 1.0
        alpha_prev = 0.0
        f_prev = f0
        g0_dot_d = np.dot(g0, d)
        
        for i in range(max_iter):
            self._set_flat_params(x0 + alpha * d)
            f = float(closure())
            g = self._gather_flat_grad()
            g_dot_d = np.dot(g, d)
            
            # Armijo condition
            if f > f0 + c1 * alpha * g0_dot_d:
                return self._zoom(closure, x0, d, alpha_prev, alpha, f0, g0, g0_dot_d, c1, c2)
            
            # Curvature condition
            if abs(g_dot_d) <= -c2 * g0_dot_d:
                return alpha
            
            if g_dot_d >= 0:
                return self._zoom(closure, x0, d, alpha, alpha_prev, f0, g0, g0_dot_d, c1, c2)
            
            alpha_prev = alpha
            f_prev = f
            alpha = min(2 * alpha, 10.0)
        
        return alpha
    
    def _zoom(self, closure, x0, d, alpha_lo, alpha_hi, f0, g0, g0_dot_d, c1, c2) -> float:
        """Zoom phase of strong Wolfe line search."""
        for _ in range(10):
            alpha = 0.5 * (alpha_lo + alpha_hi)
            self._set_flat_params(x0 + alpha * d)
            f = float(closure())
            g = self._gather_flat_grad()
            g_dot_d = np.dot(g, d)
            
            if f > f0 + c1 * alpha * g0_dot_d:
                alpha_hi = alpha
            else:
                if abs(g_dot_d) <= -c2 * g0_dot_d:
                    return alpha
                if g_dot_d * (alpha_hi - alpha_lo) >= 0:
                    alpha_hi = alpha_lo
                alpha_lo = alpha
        
        return alpha


# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.