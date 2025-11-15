import numpy as np
from tensor import Tensor

def relu(x):
    return x.maximum(Tensor(0.0, dtype=x.dtype))

def sigmoid(x):
    return 1 / (1 + (-x).exp())

def tanh(x):
    e_pos = x.exp()
    e_neg = (-x).exp()
    return (e_pos - e_neg)/(e_pos + e_neg)

def softmax(x, axis=-1):
    x_stable = x - x.max(axis=axis, keepdims=True)
    e = x_stable.exp()
    e_sum = e.sum(axis=axis, keepdims=True)
    return e / e_sum