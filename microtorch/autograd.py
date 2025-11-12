import numpy as np
from tensor import Tensor

class Operation:
    def __init__(self, *parents, **attributes):
        self.parents = parents                      # reference to input tensors
        self.forward_cache = ()                     # cached arrays for backward computation
        self.attributes = attributes                # static operation attributes (e.g. axis, shape)

    def cache_for_backward(self, *xs):
        self.forward_cache = xs

class Add(Operation):
    def forward(self, a, b):
        return a + b
    
    def backward(self, upstream_grad):
        return upstream_grad, upstream_grad
    
class Mul(Operation):
    def forward(self, a, b):
        self.cache_for_backward(a, b)
        return a * b
    
    def backward(self, upstream_grad):
        a, b = self.forward_cache
        grad_a = upstream_grad * b
        grad_b = upstream_grad * a
        return grad_a, grad_b

def apply(operation, *parents, **attributes):
    op = operation(*parents, **attributes)                              # creates operation object
    input = [tensor.data for tensor in parents]                         # extract operation inputs from parent tensors as raw numpy arrays
    requires_grad = any(tensor.requires_grad for tensor in parents)     # determine if any parent requires a gradient
    output = Tensor(op.forward(*input), requires_grad=requires_grad)    # wrap result in a tensor
    if requires_grad:
        output.grad_fn = op # store operation that created output tensor
    return output