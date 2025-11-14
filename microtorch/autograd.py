import numpy as np

class Operation:
    def __init__(self, *parents, **attributes):
        self.parents = parents                      # reference to input tensors
        self.forward_cache = ()                     # cached arrays for backward computation
        self.attributes = attributes                # static operation attributes (e.g. axis, shape)

    def cache_for_backward(self, *xs):
        self.forward_cache = xs

def sum_to_shape(grad, shape):
    shape = tuple(shape)
    if grad.shape == shape:
        return grad
    
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)

    for i, (gdim, sdim) in enumerate(zip(grad.shape, shape)):
        if sdim == 1 and gdim != 1:
            grad = grad.sum(axis=i, keepdim=True)

    return grad.reshape(shape)

class Add(Operation):
    def forward(self, a, b):
        return a + b
    
    def backward(self, upstream_grad):
        a_tensor, b_tensor = self.parents
        a_shape = a_tensor.data.shape
        b_shape = b_tensor.data.shape
        grad_a = sum_to_shape(upstream_grad, a_shape)
        grad_b = sum_to_shape(upstream_grad, b_shape)
        return grad_a, grad_b
    
class Mul(Operation):
    def forward(self, a, b):
        self.cache_for_backward(a, b)
        return a * b
    
    def backward(self, upstream_grad):
        a, b = self.forward_cache
        grad_a = upstream_grad * b
        grad_b = upstream_grad * a
        return grad_a, grad_b
    
class MatMul(Operation):
    def forward(self, a, b):
        self.cache_for_backward(a, b)
        return a @ b
    
    def backward(self, upstream_grad):
        a, b = self.forward_cache
        grad_a = upstream_grad @ b.T
        grad_b = a.T @ upstream_grad    # derivative w.r.t. b before upstream_grad because matrix multiplication is not commutative
        return grad_a, grad_b
    
class Sum(Operation):
    def forward(self, a):
        self.attributes = a.shape       # store shape as attribute
        return a.sum(),
    
    def backward(self, upstream_grad):
        return (np.ones(self.attributes, dtype=upstream_grad.dtype) * upstream_grad,) # create matrix of original size with copies of upstream gradients
    
class ReLU(Operation):
    def forward(self, a):
        self.cache_for_backward(a)
        return np.maximum(0, a)
    
    def backward(self, upstream_grad):
        (a,) = self.forward_cache       # unpack 1-tuple
        return (upstream_grad * (a > 0),)
    
class Neg(Operation):
    def forward(self, a):
        return -a
    
    def backward(self, upstream_grad):
        return (- upstream_grad,)

