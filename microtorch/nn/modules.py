import numpy as np
from ..tensor import Tensor

class Module:
    def parameters(self):
        return []
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

class Linear(Module):
    def __init__(self, in_features, out_features):
        w = np.random.randn(out_features, in_features).astype(np.float32)
        b = np.zeros(out_features, dtype=np.float32)

        self.w = Tensor(w, requires_grad=True)
        self.b = Tensor(b, requires_grad=True)

    def __call__(self, x):
        return x @ self.w.T + self.b
    
    def parameters(self):
        return [self.w, self.b]