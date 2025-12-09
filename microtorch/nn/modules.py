from .. import autograd as ag
from ..tensor import apply, Tensor
import numpy as np

class Module:
    def parameters(self):
        return []
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

class Linear(Module):
    def __init__(self, in_features, out_features):
        bound = 1.0 / np.sqrt(in_features)

        w = np.random.uniform(
            low=-bound,
            high=bound,
            size=(out_features, in_features)
        ).astype(np.float32)


        b = np.random.uniform(
            low=-bound,
            high=bound,
            size=(out_features,)
        ).astype(np.float32)

        self.w = Tensor(w, requires_grad=True)
        self.b = Tensor(b, requires_grad=True)

    def __call__(self, x):
        return x @ self.w.T + self.b
    
    def parameters(self):
        return [self.w, self.b]
    
class Conv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # kaiming initialization of weights
        fan_in = in_channels * kernel_size**2
        bound = np.sqrt(2.0 / fan_in)
        w = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * bound

        # initialize biases as 0
        b = np.zeros((1, out_channels, 1, 1), dtype=np.float32)

        # wrap parameters in Tensor object
        self.w = Tensor(w.astype(np.float32), requires_grad=True)
        self.b = Tensor(b.astype(np.float32), requires_grad=True)


    def __call__(self, x):
        return apply(ag.Conv2D, x, self.w, stride=self.stride, padding=self.padding) + self.b
    
    def parameters(self):
        return [self.w, self.b]