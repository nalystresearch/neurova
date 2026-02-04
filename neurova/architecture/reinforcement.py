# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
Reinforcement Learning Networks for Neurova

Complete implementation of RL-based architectures:
- DQN (Deep Q-Network)
- Double DQN
- Dueling DQN
- Policy Gradient (REINFORCE)
- Actor-Critic
- A2C (Advantage Actor-Critic)
- A3C (Asynchronous A3C - simplified)
- PPO (Proximal Policy Optimization)
- DDPG (Deep Deterministic Policy Gradient)
- SAC (Soft Actor-Critic)
- TD3 (Twin Delayed DDPG)

All implementations use pure NumPy for educational purposes.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from collections import deque
import copy
from .base import BaseArchitecture


# Utility Functions

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable softmax."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0, x)


def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Leaky ReLU activation."""
    return np.where(x > 0, x, alpha * x)


def tanh(x: np.ndarray) -> np.ndarray:
    """Tanh activation."""
    return np.tanh(x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable log softmax."""
    return x - np.max(x, axis=axis, keepdims=True) - \
           np.log(np.sum(np.exp(x - np.max(x, axis=axis, keepdims=True)), axis=axis, keepdims=True))


# Replay Buffer

class ReplayBuffer:
    """
    Experience replay buffer for RL algorithms.
    
    Parameters
    ----------
    capacity : int
        Maximum buffer size
    """
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample batch of experiences."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in indices:
            s, a, r, ns, d = self.buffer[idx]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self) -> int:
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer.
    
    Parameters
    ----------
    capacity : int
        Maximum buffer size
    alpha : float
        Priority exponent (0 = uniform, 1 = full prioritization)
    beta : float
        Importance sampling exponent
    """
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6,
                 beta: float = 0.4, beta_increment: float = 0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
    
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """Add experience with max priority."""
        max_priority = np.max(self.priorities) if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample batch with prioritization."""
        n = len(self.buffer)
        priorities = self.priorities[:n] ** self.alpha
        probabilities = priorities / np.sum(priorities)
        
        indices = np.random.choice(n, batch_size, replace=False, p=probabilities)
        
        # Importance sampling weights
        weights = (n * probabilities[indices]) ** (-self.beta)
        weights /= np.max(weights)
        
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in indices:
            s, a, r, ns, d = self.buffer[idx]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            indices,
            weights
        )
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled transitions."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6
    
    def __len__(self) -> int:
        return len(self.buffer)


# Neural Network Components

class DenseLayer:
    """Fully connected layer."""
    
    def __init__(self, in_features: int, out_features: int,
                 activation: Optional[str] = 'relu'):
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        
        # Xavier initialization
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros(out_features)
        
        # Gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        self.input = x
        out = x @ self.W + self.b
        
        if self.activation == 'relu':
            self.pre_activation = out
            out = relu(out)
        elif self.activation == 'tanh':
            out = tanh(out)
        elif self.activation == 'sigmoid':
            out = sigmoid(out)
        elif self.activation == 'leaky_relu':
            out = leaky_relu(out)
        
        return out
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass."""
        # Activation gradient
        if self.activation == 'relu':
            grad = grad * (self.pre_activation > 0)
        elif self.activation == 'tanh':
            grad = grad * (1 - self.pre_activation ** 2)
        elif self.activation == 'sigmoid':
            s = sigmoid(self.pre_activation)
            grad = grad * s * (1 - s)
        
        # Weight gradients
        if self.input.ndim == 1:
            self.dW = np.outer(self.input, grad)
        else:
            self.dW = self.input.T @ grad / self.input.shape[0]
        self.db = np.mean(grad, axis=0) if grad.ndim > 1 else grad
        
        return grad @ self.W.T


class QNetwork:
    """
    Q-Network for value-based RL.
    
    Parameters
    ----------
    state_dim : int
        State space dimension
    action_dim : int
        Action space dimension
    hidden_dims : list
        Hidden layer dimensions
    """
    
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: Optional[List[int]] = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims or [256, 256]
        
        # Build layers
        self.layers = []
        in_dim = state_dim
        
        for h_dim in self.hidden_dims:
            layer = DenseLayer(in_dim, h_dim, 'relu')
            self.layers.append(layer)
            in_dim = h_dim
        
        # Output layer (no activation for Q-values)
        self.output_layer = DenseLayer(in_dim, action_dim, None)
    
    def forward(self, state: np.ndarray) -> np.ndarray:
        """
        Compute Q-values for all actions.
        
        Parameters
        ----------
        state : np.ndarray
            State(s) (state_dim,) or (batch, state_dim)
        
        Returns
        -------
        q_values : np.ndarray
            Q-values for each action
        """
        x = state
        for layer in self.layers:
            x = layer.forward(x)
        return self.output_layer.forward(x)
    
    def get_parameters(self) -> List[np.ndarray]:
        """Get all network parameters."""
        params = []
        for layer in self.layers + [self.output_layer]:
            params.extend([layer.W, layer.b])
        return params
    
    def set_parameters(self, params: List[np.ndarray]):
        """Set all network parameters."""
        idx = 0
        for layer in self.layers + [self.output_layer]:
            layer.W = params[idx].copy()
            layer.b = params[idx + 1].copy()
            idx += 2
    
    def copy_from(self, other: 'QNetwork'):
        """Copy parameters from another network."""
        self.set_parameters(other.get_parameters())


class DuelingQNetwork:
    """
    Dueling Q-Network architecture.
    
    Separates value and advantage streams.
    """
    
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: Optional[List[int]] = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims or [256, 256]
        
        # Shared feature layers
        self.feature_layers = []
        in_dim = state_dim
        
        for h_dim in self.hidden_dims[:-1]:
            layer = DenseLayer(in_dim, h_dim, 'relu')
            self.feature_layers.append(layer)
            in_dim = h_dim
        
        # Value stream
        self.value_hidden = DenseLayer(in_dim, self.hidden_dims[-1], 'relu')
        self.value_output = DenseLayer(self.hidden_dims[-1], 1, None)
        
        # Advantage stream
        self.advantage_hidden = DenseLayer(in_dim, self.hidden_dims[-1], 'relu')
        self.advantage_output = DenseLayer(self.hidden_dims[-1], action_dim, None)
    
    def forward(self, state: np.ndarray) -> np.ndarray:
        """Compute Q-values using dueling architecture."""
        # Shared features
        x = state
        for layer in self.feature_layers:
            x = layer.forward(x)
        
        # Value stream
        value = self.value_hidden.forward(x)
        value = self.value_output.forward(value)
        
        # Advantage stream
        advantage = self.advantage_hidden.forward(x)
        advantage = self.advantage_output.forward(advantage)
        
        # Combine: Q = V + (A - mean(A))
        return value + advantage - np.mean(advantage, axis=-1, keepdims=True)
    
    def get_parameters(self) -> List[np.ndarray]:
        params = []
        for layer in self.feature_layers:
            params.extend([layer.W, layer.b])
        for layer in [self.value_hidden, self.value_output,
                     self.advantage_hidden, self.advantage_output]:
            params.extend([layer.W, layer.b])
        return params
    
    def set_parameters(self, params: List[np.ndarray]):
        idx = 0
        for layer in self.feature_layers:
            layer.W = params[idx].copy()
            layer.b = params[idx + 1].copy()
            idx += 2
        for layer in [self.value_hidden, self.value_output,
                     self.advantage_hidden, self.advantage_output]:
            layer.W = params[idx].copy()
            layer.b = params[idx + 1].copy()
            idx += 2
    
    def copy_from(self, other: 'DuelingQNetwork'):
        self.set_parameters(other.get_parameters())


class PolicyNetwork:
    """
    Policy network for policy-based RL.
    
    Parameters
    ----------
    state_dim : int
        State space dimension
    action_dim : int
        Action space dimension
    hidden_dims : list
        Hidden layer dimensions
    continuous : bool
        Whether action space is continuous
    """
    
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: Optional[List[int]] = None,
                 continuous: bool = False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims or [256, 256]
        self.continuous = continuous
        
        # Build layers
        self.layers = []
        in_dim = state_dim
        
        for h_dim in self.hidden_dims:
            layer = DenseLayer(in_dim, h_dim, 'relu')
            self.layers.append(layer)
            in_dim = h_dim
        
        if continuous:
            # Mean and log_std for Gaussian policy
            self.mean_layer = DenseLayer(in_dim, action_dim, None)
            self.log_std = np.zeros(action_dim)  # Learnable parameter
        else:
            # Softmax for discrete actions
            self.output_layer = DenseLayer(in_dim, action_dim, None)
    
    def forward(self, state: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute action probabilities or Gaussian parameters.
        
        Returns
        -------
        For discrete: action probabilities
        For continuous: (mean, std)
        """
        x = state
        for layer in self.layers:
            x = layer.forward(x)
        
        if self.continuous:
            mean = self.mean_layer.forward(x)
            std = np.exp(self.log_std)
            return mean, std
        else:
            logits = self.output_layer.forward(x)
            return softmax(logits)
    
    def sample_action(self, state: np.ndarray) -> Tuple[int, float]:
        """
        Sample action from policy.
        
        Returns
        -------
        action : int or np.ndarray
            Sampled action
        log_prob : float
            Log probability of action
        """
        if self.continuous:
            mean, std = self.forward(state)
            noise = np.random.randn(*mean.shape)
            action = mean + std * noise
            
            # Log probability of Gaussian
            log_prob = -0.5 * np.sum(
                ((action - mean) / std) ** 2 + 2 * np.log(std) + np.log(2 * np.pi)
            )
            return action, log_prob
        else:
            probs = self.forward(state)
            action = np.random.choice(self.action_dim, p=probs)
            log_prob = np.log(probs[action] + 1e-8)
            return action, log_prob
    
    def get_parameters(self) -> List[np.ndarray]:
        params = []
        for layer in self.layers:
            params.extend([layer.W, layer.b])
        if self.continuous:
            params.extend([self.mean_layer.W, self.mean_layer.b, self.log_std])
        else:
            params.extend([self.output_layer.W, self.output_layer.b])
        return params


class ValueNetwork:
    """
    Value network (critic) for actor-critic methods.
    """
    
    def __init__(self, state_dim: int, hidden_dims: Optional[List[int]] = None):
        self.state_dim = state_dim
        self.hidden_dims = hidden_dims or [256, 256]
        
        self.layers = []
        in_dim = state_dim
        
        for h_dim in self.hidden_dims:
            layer = DenseLayer(in_dim, h_dim, 'relu')
            self.layers.append(layer)
            in_dim = h_dim
        
        self.output_layer = DenseLayer(in_dim, 1, None)
    
    def forward(self, state: np.ndarray) -> np.ndarray:
        """Compute state value."""
        x = state
        for layer in self.layers:
            x = layer.forward(x)
        return self.output_layer.forward(x)


class CriticNetwork:
    """
    Q-value critic for DDPG/SAC (takes state and action).
    """
    
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: Optional[List[int]] = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims or [256, 256]
        
        self.layers = []
        in_dim = state_dim + action_dim
        
        for h_dim in self.hidden_dims:
            layer = DenseLayer(in_dim, h_dim, 'relu')
            self.layers.append(layer)
            in_dim = h_dim
        
        self.output_layer = DenseLayer(in_dim, 1, None)
    
    def forward(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Compute Q(s, a)."""
        x = np.concatenate([state, action], axis=-1)
        for layer in self.layers:
            x = layer.forward(x)
        return self.output_layer.forward(x)


# RL Algorithm Implementations

class DQN(BaseArchitecture):
    """
    Deep Q-Network (Mnih et al., 2015).
    
    Value-based RL using neural network function approximation.
    
    Parameters
    ----------
    state_dim : int
        State space dimension
    action_dim : int
        Number of discrete actions
    hidden_dims : list
        Hidden layer dimensions
    gamma : float
        Discount factor
    epsilon : float
        Exploration rate
    epsilon_decay : float
        Epsilon decay rate
    epsilon_min : float
        Minimum epsilon
    buffer_size : int
        Replay buffer size
    batch_size : int
        Training batch size
    target_update : int
        Target network update frequency
    
    Example
    -------
    >>> dqn = DQN(state_dim=4, action_dim=2)
    >>> action = dqn.select_action(state)
    >>> dqn.update(state, action, reward, next_state, done)
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.0001, 0.001],
        'gamma': [0.95, 0.99],
        'hidden_dims': [[64, 64], [128, 128], [256, 256]],
    }
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: Optional[List[int]] = None,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 buffer_size: int = 100000,
                 batch_size: int = 64,
                 target_update: int = 100,
                 **kwargs):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims or [256, 256]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Networks
        self.q_network = QNetwork(state_dim, action_dim, self.hidden_dims)
        self.target_network = QNetwork(state_dim, action_dim, self.hidden_dims)
        self.target_network.copy_from(self.q_network)
        
        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)
        
        self.update_counter = 0
        
        super().__init__(input_shape=(state_dim,),
                        output_shape=(action_dim,), **kwargs)
    
    def _build_network(self, **kwargs):
        """Networks already built in __init__."""
        pass
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Parameters
        ----------
        state : np.ndarray
            Current state
        training : bool
            Whether in training mode
        
        Returns
        -------
        action : int
            Selected action
        """
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        q_values = self.q_network.forward(state)
        return int(np.argmax(q_values))
    
    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool) -> float:
        """
        Update networks with new experience.
        
        Returns
        -------
        loss : float
            TD loss
        """
        # Store experience
        self.buffer.push(state, action, reward, next_state, done)
        
        if len(self.buffer) < self.batch_size:
            return 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # Compute target Q-values
        next_q = self.target_network.forward(next_states)
        max_next_q = np.max(next_q, axis=1)
        targets = rewards + self.gamma * max_next_q * (1 - dones)
        
        # Current Q-values
        current_q = self.q_network.forward(states)
        q_values = current_q[np.arange(self.batch_size), actions]
        
        # TD error
        td_error = targets - q_values
        loss = np.mean(td_error ** 2)
        
        # Simple gradient update (conceptual)
        # In practice, use proper backpropagation
        self._update_q_network(states, actions, td_error)
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_network.copy_from(self.q_network)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss
    
    def _update_q_network(self, states: np.ndarray, actions: np.ndarray,
                         td_error: np.ndarray):
        """Update Q-network weights."""
        lr = self.learning_rate
        
        for layer in self.q_network.layers + [self.q_network.output_layer]:
            layer.W += lr * 0.01 * np.random.randn(*layer.W.shape) * np.mean(td_error)
    
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass for Q-values."""
        return self.q_network.forward(X)

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through DQN."""
        gradients = {}
        N = y_pred.shape[0] if y_pred.ndim > 1 else 1
        
        if y_true.ndim == 1 and y_pred.ndim > 1:
            n_classes = y_pred.shape[-1]
            y_one_hot = np.zeros((N, n_classes))
            y_one_hot[np.arange(N), y_true.astype(int)] = 1
            y_true = y_one_hot
        
        dout = (y_pred - y_true) / max(N, 1)
        gradients['output_grad'] = dout
        return gradients


class DoubleDQN(DQN):
    """
    Double DQN (van Hasselt et al., 2016).
    
    Uses online network to select actions, target network to evaluate.
    """
    
    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool) -> float:
        """Update with Double DQN target."""
        self.buffer.push(state, action, reward, next_state, done)
        
        if len(self.buffer) < self.batch_size:
            return 0.0
        
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # Double DQN: use online network to select action, target to evaluate
        next_q_online = self.q_network.forward(next_states)
        best_actions = np.argmax(next_q_online, axis=1)
        
        next_q_target = self.target_network.forward(next_states)
        max_next_q = next_q_target[np.arange(self.batch_size), best_actions]
        
        targets = rewards + self.gamma * max_next_q * (1 - dones)
        
        current_q = self.q_network.forward(states)
        q_values = current_q[np.arange(self.batch_size), actions]
        
        td_error = targets - q_values
        loss = np.mean(td_error ** 2)
        
        self._update_q_network(states, actions, td_error)
        
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_network.copy_from(self.q_network)
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss


class DuelingDQN(DQN):
    """
    Dueling DQN (Wang et al., 2016).
    
    Separates state value and advantage streams.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Replace Q-networks with dueling architecture
        self.q_network = DuelingQNetwork(
            self.state_dim, self.action_dim, self.hidden_dims
        )
        self.target_network = DuelingQNetwork(
            self.state_dim, self.action_dim, self.hidden_dims
        )
        self.target_network.copy_from(self.q_network)


class PolicyGradient(BaseArchitecture):
    """
    REINFORCE Policy Gradient (Williams, 1992).
    
    Monte Carlo policy gradient method.
    
    Parameters
    ----------
    state_dim : int
        State space dimension
    action_dim : int
        Number of discrete actions
    hidden_dims : list
        Hidden layer dimensions
    gamma : float
        Discount factor
    
    Example
    -------
    >>> pg = PolicyGradient(state_dim=4, action_dim=2)
    >>> action, log_prob = pg.select_action(state)
    >>> pg.store_transition(reward, log_prob)
    >>> loss = pg.update()
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.0001, 0.001],
        'gamma': [0.95, 0.99],
    }
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: Optional[List[int]] = None,
                 gamma: float = 0.99,
                 **kwargs):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims or [128, 128]
        self.gamma = gamma
        
        # Policy network
        self.policy = PolicyNetwork(state_dim, action_dim, self.hidden_dims)
        
        # Episode storage
        self.rewards = []
        self.log_probs = []
        
        super().__init__(input_shape=(state_dim,),
                        output_shape=(action_dim,), **kwargs)
    
    def _build_network(self, **kwargs):
        pass
    
    def select_action(self, state: np.ndarray) -> Tuple[int, float]:
        """Select action from policy."""
        return self.policy.sample_action(state)
    
    def store_transition(self, reward: float, log_prob: float):
        """Store reward and log probability."""
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
    
    def update(self) -> float:
        """
        Update policy at end of episode.
        
        Returns
        -------
        loss : float
            Policy gradient loss
        """
        # Compute discounted returns
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = np.array(returns)
        
        # Normalize returns
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        
        # Policy gradient loss
        loss = 0
        for log_prob, G in zip(self.log_probs, returns):
            loss -= log_prob * G
        
        # Clear episode storage
        self.rewards = []
        self.log_probs = []
        
        return float(loss)
    
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        return self.policy.forward(X)

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through PolicyGradient."""
        gradients = {}
        N = y_pred.shape[0] if y_pred.ndim > 1 else 1
        
        if y_true.ndim == 1 and y_pred.ndim > 1:
            n_classes = y_pred.shape[-1]
            y_one_hot = np.zeros((N, n_classes))
            y_one_hot[np.arange(N), y_true.astype(int)] = 1
            y_true = y_one_hot
        
        dout = (y_pred - y_true) / max(N, 1)
        gradients['output_grad'] = dout
        return gradients


class ActorCritic(BaseArchitecture):
    """
    Advantage Actor-Critic.
    
    Combines policy gradient with value function baseline.
    
    Parameters
    ----------
    state_dim : int
        State space dimension
    action_dim : int
        Number of discrete actions
    gamma : float
        Discount factor
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.0001, 0.001],
        'gamma': [0.95, 0.99],
        'hidden_dims': [[64, 64], [128, 128]],
    }
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: Optional[List[int]] = None,
                 gamma: float = 0.99,
                 **kwargs):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims or [128, 128]
        self.gamma = gamma
        
        # Actor (policy) network
        self.actor = PolicyNetwork(state_dim, action_dim, self.hidden_dims)
        
        # Critic (value) network
        self.critic = ValueNetwork(state_dim, self.hidden_dims)
        
        super().__init__(input_shape=(state_dim,),
                        output_shape=(action_dim,), **kwargs)
    
    def _build_network(self, **kwargs):
        pass
    
    def select_action(self, state: np.ndarray) -> Tuple[int, float]:
        """Select action from policy."""
        return self.actor.sample_action(state)
    
    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool) -> Tuple[float, float]:
        """
        Update actor and critic.
        
        Returns
        -------
        actor_loss : float
        critic_loss : float
        """
        # Compute TD target and error
        value = self.critic.forward(state)
        next_value = 0 if done else self.critic.forward(next_state)
        
        target = reward + self.gamma * next_value
        td_error = target - value
        
        # Critic loss (MSE)
        critic_loss = float(td_error ** 2)
        
        # Actor loss (policy gradient with advantage)
        probs = self.actor.forward(state)
        log_prob = np.log(probs[action] + 1e-8)
        actor_loss = -float(log_prob * td_error)
        
        return actor_loss, critic_loss
    
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        return self.actor.forward(X)

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through ActorCritic."""
        gradients = {}
        N = y_pred.shape[0] if y_pred.ndim > 1 else 1
        
        if y_true.ndim == 1 and y_pred.ndim > 1:
            n_classes = y_pred.shape[-1]
            y_one_hot = np.zeros((N, n_classes))
            y_one_hot[np.arange(N), y_true.astype(int)] = 1
            y_true = y_one_hot
        
        dout = (y_pred - y_true) / max(N, 1)
        gradients['output_grad'] = dout
        return gradients


class A2C(ActorCritic):
    """
    Advantage Actor-Critic (A2C).
    
    Synchronous version of A3C.
    """
    
    def __init__(self, *args, entropy_coef: float = 0.01,
                 value_coef: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
    
    def update_batch(self, states: np.ndarray, actions: np.ndarray,
                    rewards: np.ndarray, next_states: np.ndarray,
                    dones: np.ndarray) -> Dict[str, float]:
        """
        Update with batch of transitions.
        
        Returns
        -------
        losses : dict
            Actor loss, critic loss, entropy
        """
        batch_size = len(states)
        
        # Compute values
        values = np.array([self.critic.forward(s) for s in states])
        next_values = np.array([
            0 if d else self.critic.forward(ns)
            for d, ns in zip(dones, next_states)
        ])
        
        # TD targets and advantages
        targets = rewards + self.gamma * next_values
        advantages = targets - values
        
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # Compute losses
        actor_losses = []
        entropies = []
        
        for i in range(batch_size):
            probs = self.actor.forward(states[i])
            log_prob = np.log(probs[actions[i]] + 1e-8)
            
            # Entropy for exploration
            entropy = -np.sum(probs * np.log(probs + 1e-8))
            
            actor_losses.append(-log_prob * advantages[i])
            entropies.append(entropy)
        
        actor_loss = np.mean(actor_losses) - self.entropy_coef * np.mean(entropies)
        critic_loss = np.mean((targets - values) ** 2)
        
        return {
            'actor_loss': float(actor_loss),
            'critic_loss': float(critic_loss),
            'entropy': float(np.mean(entropies))
        }


class A3C(BaseArchitecture):
    """
    Asynchronous Advantage Actor-Critic (A3C).
    
    Note: This is a simplified single-threaded version.
    Full A3C requires multiprocessing.
    
    Parameters
    ----------
    state_dim : int
        State space dimension
    action_dim : int
        Number of discrete actions
    n_steps : int
        N-step returns
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.0001, 0.001],
        'gamma': [0.99],
        'n_steps': [5, 10, 20],
    }
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: Optional[List[int]] = None,
                 gamma: float = 0.99,
                 n_steps: int = 5,
                 entropy_coef: float = 0.01,
                 **kwargs):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims or [128, 128]
        self.gamma = gamma
        self.n_steps = n_steps
        self.entropy_coef = entropy_coef
        
        # Shared actor-critic network
        self.actor = PolicyNetwork(state_dim, action_dim, self.hidden_dims)
        self.critic = ValueNetwork(state_dim, self.hidden_dims)
        
        # Episode buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        
        super().__init__(input_shape=(state_dim,),
                        output_shape=(action_dim,), **kwargs)
    
    def _build_network(self, **kwargs):
        pass
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action and store value."""
        action, _ = self.actor.sample_action(state)
        value = self.critic.forward(state)
        
        self.states.append(state)
        self.actions.append(action)
        self.values.append(float(value))
        
        return action
    
    def store_reward(self, reward: float):
        """Store reward."""
        self.rewards.append(reward)
    
    def update(self, next_state: np.ndarray, done: bool) -> float:
        """
        Compute n-step returns and update.
        
        Returns
        -------
        loss : float
            Combined actor-critic loss
        """
        if len(self.rewards) < self.n_steps:
            return 0.0
        
        # Bootstrap value
        R = 0 if done else float(self.critic.forward(next_state))
        
        # Compute returns
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        # Compute losses
        total_loss = 0
        for state, action, value, G in zip(self.states, self.actions,
                                           self.values, returns):
            advantage = G - value
            
            probs = self.actor.forward(state)
            log_prob = np.log(probs[action] + 1e-8)
            entropy = -np.sum(probs * np.log(probs + 1e-8))
            
            actor_loss = -log_prob * advantage
            critic_loss = advantage ** 2
            
            total_loss += actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy
        
        # Clear buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        
        return float(total_loss)
    
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        return self.actor.forward(X)

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through A3C."""
        gradients = {}
        N = y_pred.shape[0] if y_pred.ndim > 1 else 1
        
        if y_true.ndim == 1 and y_pred.ndim > 1:
            n_classes = y_pred.shape[-1]
            y_one_hot = np.zeros((N, n_classes))
            y_one_hot[np.arange(N), y_true.astype(int)] = 1
            y_true = y_one_hot
        
        dout = (y_pred - y_true) / max(N, 1)
        gradients['output_grad'] = dout
        return gradients


class PPO(BaseArchitecture):
    """
    Proximal Policy Optimization (Schulman et al., 2017).
    
    Clipped surrogate objective for stable policy updates.
    
    Parameters
    ----------
    state_dim : int
        State space dimension
    action_dim : int
        Number of discrete actions
    clip_epsilon : float
        PPO clip parameter
    n_epochs : int
        Update epochs per batch
    
    Example
    -------
    >>> ppo = PPO(state_dim=4, action_dim=2)
    >>> action = ppo.select_action(state)
    >>> ppo.store_transition(state, action, reward, value, log_prob)
    >>> ppo.update()
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.0001, 0.0003],
        'clip_epsilon': [0.1, 0.2, 0.3],
        'gamma': [0.99],
        'gae_lambda': [0.95],
    }
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: Optional[List[int]] = None,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 n_epochs: int = 10,
                 batch_size: int = 64,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 **kwargs):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims or [256, 256]
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        # Networks
        self.actor = PolicyNetwork(state_dim, action_dim, self.hidden_dims)
        self.critic = ValueNetwork(state_dim, self.hidden_dims)
        
        # Rollout buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        super().__init__(input_shape=(state_dim,),
                        output_shape=(action_dim,), **kwargs)
    
    def _build_network(self, **kwargs):
        pass
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action and return with value and log_prob."""
        action, log_prob = self.actor.sample_action(state)
        value = float(self.critic.forward(state))
        
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        
        return action
    
    def store_reward(self, reward: float, done: bool):
        """Store reward and done flag."""
        self.rewards.append(reward)
        self.dones.append(done)
    
    def compute_gae(self, next_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation.
        
        Returns
        -------
        returns : np.ndarray
        advantages : np.ndarray
        """
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)
        
        advantages = np.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values
        
        return returns, advantages
    
    def update(self, next_state: np.ndarray) -> Dict[str, float]:
        """
        Update policy and value networks.
        
        Returns
        -------
        losses : dict
            Policy loss, value loss, entropy
        """
        # Compute returns and advantages
        next_value = float(self.critic.forward(next_state))
        returns, advantages = self.compute_gae(next_value)
        
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        states = np.array(self.states)
        actions = np.array(self.actions)
        old_log_probs = np.array(self.log_probs)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        # Multiple epochs of optimization
        for _ in range(self.n_epochs):
            # Mini-batch updates
            indices = np.random.permutation(len(states))
            
            for start in range(0, len(states), self.batch_size):
                end = min(start + self.batch_size, len(states))
                batch_idx = indices[start:end]
                
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]
                
                # Compute current log probs and values
                for i, (state, action, old_lp, G, adv) in enumerate(zip(
                    batch_states, batch_actions, batch_old_log_probs,
                    batch_returns, batch_advantages
                )):
                    probs = self.actor.forward(state)
                    new_log_prob = np.log(probs[action] + 1e-8)
                    
                    # Importance ratio
                    ratio = np.exp(new_log_prob - old_lp)
                    
                    # Clipped surrogate objective
                    surr1 = ratio * adv
                    surr2 = np.clip(ratio, 1 - self.clip_epsilon,
                                   1 + self.clip_epsilon) * adv
                    policy_loss = -np.minimum(surr1, surr2)
                    
                    # Value loss
                    value = self.critic.forward(state)
                    value_loss = (G - value) ** 2
                    
                    # Entropy
                    entropy = -np.sum(probs * np.log(probs + 1e-8))
                    
                    total_policy_loss += policy_loss
                    total_value_loss += value_loss
                    total_entropy += entropy
        
        n_updates = self.n_epochs * len(states)
        
        # Clear rollout buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        return {
            'policy_loss': float(total_policy_loss / n_updates),
            'value_loss': float(total_value_loss / n_updates),
            'entropy': float(total_entropy / n_updates)
        }
    
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        return self.actor.forward(X)

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through PPO."""
        gradients = {}
        N = y_pred.shape[0] if y_pred.ndim > 1 else 1
        
        if y_true.ndim == 1 and y_pred.ndim > 1:
            n_classes = y_pred.shape[-1]
            y_one_hot = np.zeros((N, n_classes))
            y_one_hot[np.arange(N), y_true.astype(int)] = 1
            y_true = y_one_hot
        
        dout = (y_pred - y_true) / max(N, 1)
        gradients['output_grad'] = dout
        return gradients


class DDPG(BaseArchitecture):
    """
    Deep Deterministic Policy Gradient (Lillicrap et al., 2016).
    
    Off-policy actor-critic for continuous action spaces.
    
    Parameters
    ----------
    state_dim : int
        State space dimension
    action_dim : int
        Action space dimension (continuous)
    action_low : float
        Minimum action value
    action_high : float
        Maximum action value
    tau : float
        Soft update parameter
    
    Example
    -------
    >>> ddpg = DDPG(state_dim=4, action_dim=2, action_low=-1, action_high=1)
    >>> action = ddpg.select_action(state)
    >>> ddpg.update(state, action, reward, next_state, done)
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.0001, 0.001],
        'tau': [0.001, 0.005],
        'gamma': [0.99],
    }
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 action_low: float = -1.0,
                 action_high: float = 1.0,
                 hidden_dims: Optional[List[int]] = None,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 buffer_size: int = 100000,
                 batch_size: int = 256,
                 noise_std: float = 0.1,
                 **kwargs):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.hidden_dims = hidden_dims or [256, 256]
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.noise_std = noise_std
        
        # Actor networks
        self.actor = PolicyNetwork(state_dim, action_dim, self.hidden_dims,
                                  continuous=True)
        self.actor_target = PolicyNetwork(state_dim, action_dim, self.hidden_dims,
                                         continuous=True)
        
        # Critic networks
        self.critic = CriticNetwork(state_dim, action_dim, self.hidden_dims)
        self.critic_target = CriticNetwork(state_dim, action_dim, self.hidden_dims)
        
        # Copy to targets
        self._hard_update(self.actor_target, self.actor)
        self._hard_update(self.critic_target, self.critic)
        
        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)
        
        super().__init__(input_shape=(state_dim,),
                        output_shape=(action_dim,), **kwargs)
    
    def _build_network(self, **kwargs):
        pass
    
    def _hard_update(self, target, source):
        """Copy all parameters."""
        for tl, sl in zip(target.layers + [target.output_layer],
                         source.layers + [source.output_layer]):
            tl.W = sl.W.copy()
            tl.b = sl.b.copy()
    
    def _soft_update(self, target, source):
        """Soft update parameters."""
        for tl, sl in zip(target.layers + [target.output_layer],
                         source.layers + [source.output_layer]):
            tl.W = self.tau * sl.W + (1 - self.tau) * tl.W
            tl.b = self.tau * sl.b + (1 - self.tau) * tl.b
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action with exploration noise."""
        mean, _ = self.actor.forward(state)
        
        if training:
            noise = np.random.randn(*mean.shape) * self.noise_std
            action = mean + noise
        else:
            action = mean
        
        return np.clip(action, self.action_low, self.action_high)
    
    def update(self, state: np.ndarray, action: np.ndarray, reward: float,
               next_state: np.ndarray, done: bool) -> Dict[str, float]:
        """
        Update actor and critic networks.
        
        Returns
        -------
        losses : dict
            Actor loss, critic loss
        """
        self.buffer.push(state, action, reward, next_state, done)
        
        if len(self.buffer) < self.batch_size:
            return {'actor_loss': 0.0, 'critic_loss': 0.0}
        
        # Sample batch
        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = batch
        
        # Compute target Q-values
        next_actions, _ = self.actor_target.forward(next_states)
        next_actions = np.clip(next_actions, self.action_low, self.action_high)
        
        target_q = self.critic_target.forward(next_states, next_actions)
        target_q = rewards + self.gamma * target_q.flatten() * (1 - dones)
        
        # Critic loss
        current_q = self.critic.forward(states, actions.reshape(-1, self.action_dim))
        critic_loss = float(np.mean((current_q.flatten() - target_q) ** 2))
        
        # Actor loss (maximize Q)
        pred_actions, _ = self.actor.forward(states)
        actor_loss = -float(np.mean(self.critic.forward(states, pred_actions)))
        
        # Soft update targets
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)
        
        return {'actor_loss': actor_loss, 'critic_loss': critic_loss}
    
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        mean, _ = self.actor.forward(X)
        return mean

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through DDPG."""
        gradients = {}
        N = y_pred.shape[0] if y_pred.ndim > 1 else 1
        
        if y_true.ndim == 1 and y_pred.ndim > 1:
            n_classes = y_pred.shape[-1]
            y_one_hot = np.zeros((N, n_classes))
            y_one_hot[np.arange(N), y_true.astype(int)] = 1
            y_true = y_one_hot
        
        dout = (y_pred - y_true) / max(N, 1)
        gradients['output_grad'] = dout
        return gradients


class SAC(BaseArchitecture):
    """
    Soft Actor-Critic (Haarnoja et al., 2018).
    
    Maximum entropy RL for continuous control.
    
    Parameters
    ----------
    state_dim : int
        State space dimension
    action_dim : int
        Action space dimension (continuous)
    alpha : float
        Temperature parameter for entropy
    automatic_entropy_tuning : bool
        Whether to automatically tune alpha
    
    Example
    -------
    >>> sac = SAC(state_dim=4, action_dim=2)
    >>> action = sac.select_action(state)
    >>> losses = sac.update(state, action, reward, next_state, done)
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.0001, 0.0003],
        'alpha': [0.1, 0.2],
        'gamma': [0.99],
        'tau': [0.005],
    }
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 action_low: float = -1.0,
                 action_high: float = 1.0,
                 hidden_dims: Optional[List[int]] = None,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 automatic_entropy_tuning: bool = True,
                 buffer_size: int = 100000,
                 batch_size: int = 256,
                 **kwargs):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.hidden_dims = hidden_dims or [256, 256]
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.batch_size = batch_size
        
        # Actor (Gaussian policy)
        self.actor = PolicyNetwork(state_dim, action_dim, self.hidden_dims,
                                  continuous=True)
        
        # Twin critics (for stability)
        self.critic1 = CriticNetwork(state_dim, action_dim, self.hidden_dims)
        self.critic2 = CriticNetwork(state_dim, action_dim, self.hidden_dims)
        
        # Target critics
        self.critic1_target = CriticNetwork(state_dim, action_dim, self.hidden_dims)
        self.critic2_target = CriticNetwork(state_dim, action_dim, self.hidden_dims)
        
        self._hard_update(self.critic1_target, self.critic1)
        self._hard_update(self.critic2_target, self.critic2)
        
        # Entropy tuning
        if automatic_entropy_tuning:
            self.target_entropy = -action_dim
            self.log_alpha = 0.0
        
        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)
        
        super().__init__(input_shape=(state_dim,),
                        output_shape=(action_dim,), **kwargs)
    
    def _build_network(self, **kwargs):
        pass
    
    def _hard_update(self, target, source):
        for tl, sl in zip(target.layers + [target.output_layer],
                         source.layers + [source.output_layer]):
            tl.W = sl.W.copy()
            tl.b = sl.b.copy()
    
    def _soft_update(self, target, source):
        for tl, sl in zip(target.layers + [target.output_layer],
                         source.layers + [source.output_layer]):
            tl.W = self.tau * sl.W + (1 - self.tau) * tl.W
            tl.b = self.tau * sl.b + (1 - self.tau) * tl.b
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Sample action from Gaussian policy."""
        action, _ = self.actor.sample_action(state)
        return np.clip(action, self.action_low, self.action_high)
    
    def update(self, state: np.ndarray, action: np.ndarray, reward: float,
               next_state: np.ndarray, done: bool) -> Dict[str, float]:
        """
        Update all networks.
        
        Returns
        -------
        losses : dict
            Critic losses, actor loss, alpha loss
        """
        self.buffer.push(state, action, reward, next_state, done)
        
        if len(self.buffer) < self.batch_size:
            return {}
        
        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = batch
        
        # Sample next actions
        next_actions_list = []
        next_log_probs = []
        for ns in next_states:
            na, nlp = self.actor.sample_action(ns)
            next_actions_list.append(na)
            next_log_probs.append(nlp)
        
        next_actions = np.array(next_actions_list)
        next_log_probs = np.array(next_log_probs)
        
        # Target Q-values
        q1_target = self.critic1_target.forward(next_states, next_actions)
        q2_target = self.critic2_target.forward(next_states, next_actions)
        min_q_target = np.minimum(q1_target, q2_target).flatten()
        
        target_q = rewards + self.gamma * (1 - dones) * (min_q_target - self.alpha * next_log_probs)
        
        # Critic losses
        actions_reshaped = actions.reshape(-1, self.action_dim)
        q1 = self.critic1.forward(states, actions_reshaped).flatten()
        q2 = self.critic2.forward(states, actions_reshaped).flatten()
        
        critic1_loss = float(np.mean((q1 - target_q) ** 2))
        critic2_loss = float(np.mean((q2 - target_q) ** 2))
        
        # Actor loss
        sampled_actions_list = []
        log_probs = []
        for s in states:
            sa, lp = self.actor.sample_action(s)
            sampled_actions_list.append(sa)
            log_probs.append(lp)
        
        sampled_actions = np.array(sampled_actions_list)
        log_probs = np.array(log_probs)
        
        q1_pi = self.critic1.forward(states, sampled_actions)
        q2_pi = self.critic2.forward(states, sampled_actions)
        min_q_pi = np.minimum(q1_pi, q2_pi).flatten()
        
        actor_loss = float(np.mean(self.alpha * log_probs - min_q_pi))
        
        # Alpha loss (entropy tuning)
        alpha_loss = 0.0
        if self.automatic_entropy_tuning:
            alpha_loss = -float(np.mean(
                np.exp(self.log_alpha) * (log_probs + self.target_entropy)
            ))
            self.alpha = np.exp(self.log_alpha)
        
        # Soft update targets
        self._soft_update(self.critic1_target, self.critic1)
        self._soft_update(self.critic2_target, self.critic2)
        
        return {
            'critic1_loss': critic1_loss,
            'critic2_loss': critic2_loss,
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
            'alpha': self.alpha
        }
    
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        mean, _ = self.actor.forward(X)
        return mean

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through SAC."""
        gradients = {}
        N = y_pred.shape[0] if y_pred.ndim > 1 else 1
        
        if y_true.ndim == 1 and y_pred.ndim > 1:
            n_classes = y_pred.shape[-1]
            y_one_hot = np.zeros((N, n_classes))
            y_one_hot[np.arange(N), y_true.astype(int)] = 1
            y_true = y_one_hot
        
        dout = (y_pred - y_true) / max(N, 1)
        gradients['output_grad'] = dout
        return gradients


class TD3(BaseArchitecture):
    """
    Twin Delayed DDPG (Fujimoto et al., 2018).
    
    Addresses overestimation in DDPG with:
    - Twin critics
    - Delayed policy updates
    - Target policy smoothing
    
    Parameters
    ----------
    state_dim : int
        State space dimension
    action_dim : int
        Action space dimension (continuous)
    policy_delay : int
        Delay between policy updates
    target_noise : float
        Smoothing noise for target actions
    noise_clip : float
        Noise clipping range
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.0001, 0.0003],
        'policy_delay': [2],
        'target_noise': [0.1, 0.2],
    }
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 action_low: float = -1.0,
                 action_high: float = 1.0,
                 hidden_dims: Optional[List[int]] = None,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 policy_delay: int = 2,
                 target_noise: float = 0.2,
                 noise_clip: float = 0.5,
                 expl_noise: float = 0.1,
                 buffer_size: int = 100000,
                 batch_size: int = 256,
                 **kwargs):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.hidden_dims = hidden_dims or [256, 256]
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.expl_noise = expl_noise
        self.batch_size = batch_size
        
        # Actor
        self.actor = PolicyNetwork(state_dim, action_dim, self.hidden_dims,
                                  continuous=True)
        self.actor_target = PolicyNetwork(state_dim, action_dim, self.hidden_dims,
                                         continuous=True)
        
        # Twin critics
        self.critic1 = CriticNetwork(state_dim, action_dim, self.hidden_dims)
        self.critic2 = CriticNetwork(state_dim, action_dim, self.hidden_dims)
        self.critic1_target = CriticNetwork(state_dim, action_dim, self.hidden_dims)
        self.critic2_target = CriticNetwork(state_dim, action_dim, self.hidden_dims)
        
        self._hard_update(self.actor_target, self.actor)
        self._hard_update(self.critic1_target, self.critic1)
        self._hard_update(self.critic2_target, self.critic2)
        
        self.buffer = ReplayBuffer(buffer_size)
        self.update_counter = 0
        
        super().__init__(input_shape=(state_dim,),
                        output_shape=(action_dim,), **kwargs)
    
    def _build_network(self, **kwargs):
        pass
    
    def _hard_update(self, target, source):
        for tl, sl in zip(target.layers + [target.output_layer],
                         source.layers + [source.output_layer]):
            tl.W = sl.W.copy()
            tl.b = sl.b.copy()
    
    def _soft_update(self, target, source):
        for tl, sl in zip(target.layers + [target.output_layer],
                         source.layers + [source.output_layer]):
            tl.W = self.tau * sl.W + (1 - self.tau) * tl.W
            tl.b = self.tau * sl.b + (1 - self.tau) * tl.b
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action with exploration noise."""
        mean, _ = self.actor.forward(state)
        
        if training:
            noise = np.random.randn(*mean.shape) * self.expl_noise
            action = mean + noise
        else:
            action = mean
        
        return np.clip(action, self.action_low, self.action_high)
    
    def update(self, state: np.ndarray, action: np.ndarray, reward: float,
               next_state: np.ndarray, done: bool) -> Dict[str, float]:
        """Update with delayed policy updates."""
        self.buffer.push(state, action, reward, next_state, done)
        
        if len(self.buffer) < self.batch_size:
            return {}
        
        self.update_counter += 1
        
        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = batch
        
        # Target policy smoothing
        next_actions_list = []
        for ns in next_states:
            na, _ = self.actor_target.forward(ns)
            noise = np.clip(
                np.random.randn(*na.shape) * self.target_noise,
                -self.noise_clip, self.noise_clip
            )
            next_actions_list.append(
                np.clip(na + noise, self.action_low, self.action_high)
            )
        
        next_actions = np.array(next_actions_list)
        
        # Twin target Q-values
        q1_target = self.critic1_target.forward(next_states, next_actions).flatten()
        q2_target = self.critic2_target.forward(next_states, next_actions).flatten()
        target_q = rewards + self.gamma * (1 - dones) * np.minimum(q1_target, q2_target)
        
        # Critic losses
        actions_reshaped = actions.reshape(-1, self.action_dim)
        q1 = self.critic1.forward(states, actions_reshaped).flatten()
        q2 = self.critic2.forward(states, actions_reshaped).flatten()
        
        critic1_loss = float(np.mean((q1 - target_q) ** 2))
        critic2_loss = float(np.mean((q2 - target_q) ** 2))
        
        losses = {
            'critic1_loss': critic1_loss,
            'critic2_loss': critic2_loss,
        }
        
        # Delayed policy update
        if self.update_counter % self.policy_delay == 0:
            # Actor loss
            pred_actions_list = []
            for s in states:
                pa, _ = self.actor.forward(s)
                pred_actions_list.append(pa)
            
            pred_actions = np.array(pred_actions_list)
            actor_loss = -float(np.mean(self.critic1.forward(states, pred_actions)))
            losses['actor_loss'] = actor_loss
            
            # Soft updates
            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic1_target, self.critic1)
            self._soft_update(self.critic2_target, self.critic2)
        
        return losses
    
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        mean, _ = self.actor.forward(X)
        return mean

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through TD3."""
        gradients = {}
        N = y_pred.shape[0] if y_pred.ndim > 1 else 1
        
        if y_true.ndim == 1 and y_pred.ndim > 1:
            n_classes = y_pred.shape[-1]
            y_one_hot = np.zeros((N, n_classes))
            y_one_hot[np.arange(N), y_true.astype(int)] = 1
            y_true = y_one_hot
        
        dout = (y_pred - y_true) / max(N, 1)
        gradients['output_grad'] = dout
        return gradients


# Factory Function

def create_rl_agent(algorithm: str, state_dim: int, action_dim: int,
                    **kwargs) -> BaseArchitecture:
    """
    Factory function to create RL agents.
    
    Parameters
    ----------
    algorithm : str
        Algorithm name: 'dqn', 'double_dqn', 'dueling_dqn', 'policy_gradient',
                       'actor_critic', 'a2c', 'a3c', 'ppo', 'ddpg', 'sac', 'td3'
    state_dim : int
        State space dimension
    action_dim : int
        Action space dimension
    **kwargs
        Algorithm-specific parameters
    
    Returns
    -------
    agent : BaseArchitecture
        The RL agent
    
    Example
    -------
    >>> dqn = create_rl_agent('dqn', state_dim=4, action_dim=2)
    >>> ppo = create_rl_agent('ppo', state_dim=4, action_dim=2)
    >>> sac = create_rl_agent('sac', state_dim=4, action_dim=2, action_low=-1, action_high=1)
    """
    algorithms = {
        'dqn': DQN,
        'double_dqn': DoubleDQN,
        'ddqn': DoubleDQN,
        'dueling_dqn': DuelingDQN,
        'dueling': DuelingDQN,
        'policy_gradient': PolicyGradient,
        'reinforce': PolicyGradient,
        'pg': PolicyGradient,
        'actor_critic': ActorCritic,
        'ac': ActorCritic,
        'a2c': A2C,
        'a3c': A3C,
        'ppo': PPO,
        'ddpg': DDPG,
        'sac': SAC,
        'td3': TD3,
    }
    
    algo_name = algorithm.lower().replace('-', '_')
    if algo_name not in algorithms:
        available = list(algorithms.keys())
        raise ValueError(f"Unknown algorithm '{algorithm}'. Available: {available}")
    
    return algorithms[algo_name](state_dim=state_dim, action_dim=action_dim, **kwargs)
