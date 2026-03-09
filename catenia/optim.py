from typing import Iterable
from abc import ABC, abstractmethod

import numpy as np

from catenia.nn import Parameter


class Optimizer(ABC):

    def __init__(self, parameters: Iterable[Parameter]):
        self.params = list(parameters)
    
    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    @abstractmethod
    def step(self): pass


class SGD(Optimizer):
    """Best Use Case: High-quality convergence on simple models."""

    def __init__(self, parameters, lr=1e-2, momentum=0.0):
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self.vs = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for p, v in zip(self.params, self.vs):
            if p.grad is None: continue

            # v = beta * v + lr * grad
            v[:] = self.momentum * v + self.lr * p.grad
            p.data -= v


class RMSProp(Optimizer):
    """Best Use Case: Recurrent Neural Networks (RNNs)."""

    def __init__(self, parameters, lr=1e-2, alpha=0.9, eps=1e-8):
        super().__init__(parameters)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.sq_avg = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for p, s in zip(self.params, self.sq_avg):
            if p.grad is None: continue
            
            # s = alpha * s + (1 - alpha) * grad^2
            s[:] = self.alpha * s + (1 - self.alpha) * (p.grad ** 2)
            p.data -= (self.lr / (np.sqrt(s) + self.eps)) * p.grad


class AdaDelta(Optimizer):
    """Best Use Case: When you don't want to tune a learning rate."""

    def __init__(self, parameters, rho=0.9, eps=1e-8):
        super().__init__(parameters)
        self.rho = rho
        self.eps = eps
        # Accumulate squared gradients
        self.sq_grad_avg = [np.zeros_like(p.data) for p in self.params]
        # Accumulate squared updates
        self.sq_upd_avg = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            
            # Update squared gradient average
            self.sq_grad_avg[i][:] = self.rho * self.sq_grad_avg[i] + (1 - self.rho) * (p.grad ** 2)
            
            # Compute update: sqrt(RMS_update_prev / RMS_grad_now) * grad
            std_grad = np.sqrt(self.sq_grad_avg[i] + self.eps)
            std_upd = np.sqrt(self.sq_upd_avg[i] + self.eps)
            update = (std_upd / std_grad) * p.grad
            
            # Apply update
            p.data -= update
            
            # Update squared update average
            self.sq_upd_avg[i][:] = self.rho * self.sq_upd_avg[i] + (1 - self.rho) * (update ** 2)


class Adam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params)
        self.lr, self.beta1, self.beta2, self.eps = lr, betas[0], betas[1], eps
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad**2)
            
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Optimizer):
    """Best Use Case: Transformers, Large CNNs, General Purpose."""

    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        super().__init__(parameters) # This sets self.params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        
        # Use self.params here to match the base class
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params): # Changed from self.parameters
            if p.grad is None:
                continue

            # Apply Weight Decay directly to data
            # This is the "W" in AdamW: p = p - lr * weight_decay * p
            if self.weight_decay != 0:
                p.data -= self.lr * self.weight_decay * p.data

            # Update Adam moments
            # m = beta1 * m + (1 - beta1) * grad
            self.m[i][:] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            # v = beta2 * v + (1 - beta2) * grad^2
            self.v[i][:] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad ** 2)

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            # Update parameters with adaptive gradient
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
