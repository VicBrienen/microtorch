import numpy as np
from tensor import Tensor

def relu(x):
    return x.maximum(Tensor(0.0, dtype=x.dtype))

def softmax(x, axis=-1):
    x_stable = x - x.max(axis=axis, keepdims=True)
    e = x_stable.exp()
    e_sum = e.sum(axis=axis, keepdims=True)
    return e / e_sum

def cross_entropy(logits, targets, axis=-1):
    probs = softmax(logits, axis=axis)
    log_probs = probs.log()
    loss_per_sample = -(targets * log_probs).sum(axis=axis)
    batch_size = logits.data.shape[0]
    return loss_per_sample.sum() / Tensor(batch_size, dtype=logits.dtype)