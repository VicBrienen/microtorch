from ..tensor import Tensor

def relu(x):
    return x * (x > 0)

def softmax(x, axis=-1):
    x_stable = x - x.max(axis=axis, keepdims=True)
    e = x_stable.exp()
    e_sum = e.sum(axis=axis, keepdims=True)
    return e / e_sum

def log_softmax(x, axis=-1):
    x_stable = x - x.max(axis=axis, keepdims=True)
    logsumexp = x_stable.exp().sum(axis=axis, keepdims=True).log()
    return x_stable - logsumexp

def cross_entropy(logits, targets, axis=-1):
    log_probs = log_softmax(logits, axis=axis)
    loss_per_sample = -(targets * log_probs).sum(axis=axis)
    batch_size = loss_per_sample.data.shape[0]
    return loss_per_sample.sum() / Tensor(batch_size, dtype=logits.dtype)