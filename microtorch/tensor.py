import numpy as np
import autograd as ag

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
        self.grad = np.array(1.0, dtype=self.dtype) # seed gradient at the root (loss)
        for tensor in reversed(topo): # reverse ordering such that we start at the loss
            operation = tensor.grad_fn # extract specific operation
            for parent_tensor, parent_grad in zip(operation.parents, operation.backward(tensor.grad)): # compute downstream gradients and loop over parent-gradient pairs

                # add the gradient
                if parent_tensor.requires_grad:
                    if parent_tensor.grad is None:
                        parent_tensor.grad = parent_grad
                    else:
                        parent_tensor.grad += parent_grad

    def __add__(self, other):
        return ag.apply(ag.Add, self, other)

    def __mul__(self, other):
        return ag.apply(ag.Mul, self, other)

    def __matmul__(self, other):
        return ag.apply(ag.MatMul, self, other)

    def __neg__(self):
        return ag.apply(ag.Neg, self)

    def __sub__(self, other):
        return ag.apply(ag.Add, self, ag.apply(ag.Neg, other))

    def sum(self):
        return ag.apply(ag.Sum, self)

    def relu(self):
        return ag.apply(ag.ReLU, self)