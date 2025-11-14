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
            grad = grad.sum(axis=i, keepdims=True)

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
    
class Sum(Operation):
    def forward(self, a):
        axis = self.attributes.get("axis", None)
        keepdims = self.attributes.get("keepdims", False)
        self.input_shape = a.shape
        self.axis = axis
        self.keepdims = keepdims
        return a.sum(axis=axis, keepdims=keepdims)
    
    def backward(self, upstream_grad):
        grad_a = np.ones(self.input_shape, dtype=upstream_grad.dtype) * upstream_grad
        return (grad_a,) # create matrix of original size with copies of upstream gradients
    
class Mul(Operation):
    def forward(self, a, b):
        self.cache_for_backward(a, b)
        return a * b
    
    def backward(self, upstream_grad):
        a, b = self.forward_cache
        grad_a = upstream_grad * b
        grad_b = upstream_grad * a
        grad_a = sum_to_shape(grad_a, a.shape)
        grad_b = sum_to_shape(grad_b, b.shape)
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
    
class Neg(Operation):
    def forward(self, a):
        return -a
    
    def backward(self, upstream_grad):
        return (- upstream_grad,)

class Pow(Operation):
    def forward(self, a):
        exponent = self.attributes["exponent"]
        self.cache_for_backward(a)
        return a ** exponent
    
    def backward(self, upstream_grad):
        (a,) = self.forward_cache
        exponent = self.attributes["exponent"]
        grad_a = upstream_grad * exponent * (a ** (exponent - 1))
        return (grad_a,)

class Exp(Operation):
    def forward(self, a):
        out = np.exp(a)
        self.cache_for_backward(out)
        return out
    
    def backward(self, upstream_grad):
        (out,) = self.forward_cache
        grad_a = upstream_grad * out
        return (grad_a,)
    
class Log(Operation):
    def forward(self, a):
        self.cache_for_backward(a)
        return np.log(a)
    
    def backward(self, upstream_grad):
        (a,) = self.forward_cache
        grad_a = upstream_grad / a
        return (grad_a,)
    
class Max(Operation):
    def forward(self, a):
        # write attributes and forward cache
        axis = self.attributes.setdefault("axis", None)
        keepdims = self.attributes.setdefault("keepdims", False)
        self.attributes["input_shape"] = a.shape
        out = np.max(a, axis=axis, keepdims=keepdims)
        self.cache_for_backward(a, out)
        return out
    
    def backward(self, upstream_grad):
        # read attributes and forward cache
        a, out = self.forward_cache
        axis = self.attributes["axis"]
        keepdims = self.attributes["keepdims"]
        input_shape = self.attributes["input_shape"]

        # broadcast max values back to input shape
        if axis is None:
            max_keep = out
        else:
            if keepdims:
                max_keep = out
            else:
                max_keep = np.expand_dims(out, axis=axis)
        max_keep = np.broadcast_to(max_keep, input_shape)

        mask = (a == max_keep) # build mask of max positions

        # broadcast upsteam gradient back to input shape
        if axis is None:
            grad_expanded = upstream_grad
        else:
            if keepdims:
                grad_expanded = upstream_grad
            else:
                grad_expanded = np.expand_dims(upstream_grad, axis=axis)
        grad_expanded = np.broadcast_to(grad_expanded, input_shape)

        # route gradients only trough max entries
        grad_a = grad_expanded * mask
        return (grad_a,)