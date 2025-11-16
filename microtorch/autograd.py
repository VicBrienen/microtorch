import numpy as np

class Operation:
    """
    Base class for atomic differentiable operations from which all other operations can be constructed.

    Args:
        *parents (Tensor): Input tensors that produced this operations output.
        **attributes: Static configuration values required by the operation.

    Stored attributes:
        parents (tuple[Tensor]): References to the input tensors.
        attributes (dict): Operation specific configuration.
        forward_cache (tuple): Values needed for gradient computation in .backward().
    """
    def __init__(self, *parents, **attributes):
        self.parents = parents
        self.forward_cache = ()
        self.attributes = attributes

    def cache_for_backward(self, *xs):
        self.forward_cache = xs

class Add(Operation):
    def forward(self, a, b):
        return a + b
    
    def backward(self, upstream_grad):
        a_tensor, b_tensor = self.parents
        a_shape, b_shape = a_tensor.data.shape, b_tensor.data.shape
        grad_a, grad_b = sum_to_shape(upstream_grad, a_shape), sum_to_shape(upstream_grad, b_shape)
        return grad_a, grad_b
    
class Mul(Operation):
    def forward(self, a, b):
        self.cache_for_backward(a, b)
        return a * b
    
    def backward(self, upstream_grad):
        a, b = self.forward_cache
        grad_a, grad_b = upstream_grad * b, upstream_grad * a
        grad_a, grad_b = sum_to_shape(grad_a, a.shape), sum_to_shape(grad_b, b.shape)
        return grad_a, grad_b

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
    
class MatMul(Operation):
    def forward(self, a, b):
        self.cache_for_backward(a, b)
        return a @ b
    
    def backward(self, upstream_grad):
        a, b = self.forward_cache
        grad_a = np.matmul(upstream_grad, np.swapaxes(b, -1, -2))
        grad_b = np.matmul(np.swapaxes(a, -1, -2), upstream_grad) # derivative w.r.t. b before upstream_grad because matrix multiplication is not commutative
        grad_a = sum_to_shape(grad_a, a.shape)
        grad_b = sum_to_shape(grad_b, b.shape)
        return grad_a, grad_b

class Transpose(Operation):
    def forward(self, a):
        return np.swapaxes(a, -1, -2)
    
    def backward(self, upstream_grad):
        return (np.swapaxes(upstream_grad, -1, -2),)
    
class Sum(Operation):
    def forward(self, a):
        axis = self.attributes.get("axis", None)
        keepdims = self.attributes.get("keepdims", False)
        self.input_shape = a.shape
        self.axis = axis
        self.keepdims = keepdims
        return a.sum(axis=axis, keepdims=keepdims)
    
    def backward(self, upstream_grad):
        axis, keepdims, input_shape = self.axis, self.keepdims, self.input_shape

        if axis is None:
            grad_a = upstream_grad
        else:
            if not keepdims:
                grad_a = np.expand_dims(upstream_grad, axis=axis)
            else:
                grad_a = upstream_grad
        grad_a = np.broadcast_to(grad_a, input_shape)
        return (grad_a,)

class Max(Operation):
    def forward(self, a):
        axis = self.attributes.setdefault("axis", None)
        keepdims = self.attributes.setdefault("keepdims", False)
        self.attributes["input_shape"] = a.shape
        out = np.max(a, axis=axis, keepdims=keepdims)
        self.cache_for_backward(a, out)
        return out
    
    def backward(self, upstream_grad):
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
        count = mask.sum(axis=axis, keepdims=True)

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
        grad_a = mask * (grad_expanded / count)
        return (grad_a,)

class Greater(Operation):
    def forward(self, a, b):
        return a > b
    
    def backward(self, upstream_grad):
        return (np.zeros_like(self.parents[0].data),
                np.zeros_like(self.parents[1].data))
    
def sum_to_shape(grad, shape):
    """
    Reduce a broadcasted gradient back to the original shape.

    Args:
    grad (np.ndarray): The upstream gradient received by a tensor
        after an operation that may have broadcast some inputs.
    shape (tuple or list): The original shape of the tensor whose
        gradient we want to compute.

    Returns:
        np.ndarray: A gradient array reduced to the original tensor shape.
    """
    shape = tuple(shape)

    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    
    for i, sdim in enumerate(shape):
        if sdim == 1 and grad.shape[i]!= 1:
            grad = grad.sum(axis=i, keepdims=True)
    
    return grad.reshape(shape)