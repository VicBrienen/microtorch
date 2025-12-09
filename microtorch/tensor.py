import numpy as np
from . import autograd as ag

class Tensor:
    def __init__(
        self, data,
        requires_grad=False,
        dtype=np.float32
        ):
        self.data = np.array(data, dtype=dtype, copy=True)                  # original numpy array (no static 1d storage)
        self.dtype = dtype                                                  # store data type
        self.requires_grad = requires_grad                                  # trainable parameters or not
        self.grad = None                                                    # tensor that stores accumulated gradients
        self.grad_fn = None                                                 # backward function or None for leaf tensor
    
    def backward(self):
        # build topological ordering
        topo, visited = [], set()
        def build(tensor):
            if tensor not in visited and tensor.grad_fn is not None:    # only traverse non leaf tensors once
                visited.add(tensor)
                for parent in tensor.grad_fn.parents:                   # recurse into parents first
                    build(parent)
                topo.append(tensor)                                     # then add this tensor to topological ordering
        build(self)                                                     # start from the tensor .backward() is called on

        # perform reverse mode automatic differentiation a.k.a. backpropagation
        self.grad = np.ones_like(self.data, dtype=self.dtype) # seed gradient at the root (loss)
        for tensor in reversed(topo): # reverse ordering such that we start at the loss
            operation = tensor.grad_fn # extract specific operation
            for parent_tensor, parent_grad in zip(operation.parents, operation.backward(tensor.grad)): # compute downstream gradients and loop over parent-gradient pairs

                # add the gradient
                if parent_tensor.requires_grad:
                    if parent_tensor.grad is None:
                        parent_tensor.grad = parent_grad
                    else:
                        parent_tensor.grad += parent_grad

    def ensure_tensor(self, other):
        if isinstance(other, Tensor):
            return other
        else:
            return  Tensor(other, dtype=self.dtype)

    def __add__(self, other):
        return apply(ag.Add, self, self.ensure_tensor(other))

    def __mul__(self, other):
        return apply(ag.Mul, self, self.ensure_tensor(other))
    
    def __truediv__(self, other):
        return self * (self.ensure_tensor(other) ** -1)
    
    def __pow__(self, exponent):
        return apply(ag.Pow, self, exponent=exponent)

    def __neg__(self):
        return apply(ag.Mul, self.ensure_tensor(-1.0), self)

    def __sub__(self, other):
        return self + (-self.ensure_tensor(other))
    
    def exp(self):
        return apply(ag.Exp, self)
    
    def log(self):
        return apply(ag.Log, self)
    
    def sum(self, axis=None, keepdims=False):
        return apply(ag.Sum, self, axis=axis, keepdims=keepdims)
    
    def __matmul__(self, other):
        return apply(ag.MatMul, self, self.ensure_tensor(other))
    
    def reshape(self, *shape):
        apply(ag.Reshape, self, shape=shape)
    
    @property
    def T(self):
        return apply(ag.Transpose, self)
    
    def max(self, axis=None, keepdims=False):
        return apply(ag.Max, self, axis=axis, keepdims=keepdims)
    
    def __gt__(self, other):
        return apply(ag.Greater, self, self.ensure_tensor(other))
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __rtruediv__(self, other):
        return self.ensure_tensor(other) * (self ** -1)
    
    def __rsub__(self, other):
        return self.ensure_tensor(other) + (-self)
    
def apply(operation, *parents, **attributes):
    op = operation(*parents, **attributes)                              # creates operation object
    input = [tensor.data for tensor in parents]                         # extract operation inputs from parent tensors as raw numpy arrays
    requires_grad = any(tensor.requires_grad for tensor in parents)     # determine if any parent requires a gradient
    output = Tensor(op.forward(*input), requires_grad=requires_grad)    # wrap result in a tensor
    if requires_grad:
        output.grad_fn = op # store operation that created output tensor
    return output