from .. import autograd as ag
from ..tensor import apply, Tensor
from . import functional as F
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
    
class MultiHeadAttention(Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0 # we stick to conventional MHA design choices where embed_dim = num_heads * head_dim

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # QKV and output projections
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

    def __call__(self, q, k, v, mask=None): # mask is not strictly necessary but heavily used in NLP and self supervised CV so it is implemented
        batch_size, seq_len, _ = q.data.shape

        Q = self.q_proj(q)
        K = self.k_proj(k)
        V = self.v_proj(v)

        # split heads and permute
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        K_T = K.permute(0, 1, 3, 2)

        scores = (Q @ K_T) / np.sqrt(self.head_dim)

        if mask is not None:
            scores = scores + mask

        attention_weights = F.softmax(scores, axis=-1)

        # multiply with value vector
        out = attention_weights @ V
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(batch_size, seq_len, self.embed_dim)

        return self.out_proj(out)
    
    def parameters(self):
        return (self.q_proj.parameters() +
                self.k_proj.parameters() +
                self.v_proj.parameters() +
                self.out_proj.parameters())