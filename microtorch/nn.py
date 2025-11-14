from tensor import Tensor
import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        w = np.random.randn(in_features, out_features).astype(np.float32)
        b = np.zeros(out_features, dtype=np.float32)

        self.w = Tensor(w, requires_grad=True)
        self.b = Tensor(b, requires_grad=True)

    def __call__(self, x):
        return x @ self.w + self.b
    
    def parameters(self):
        return [self.w, self.b]