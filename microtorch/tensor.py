import numpy as np

class Tensor:
    def __init__(
        self, data,
        requires_grad=False,
        dtype=np.float32
        ):
        array = np.array(data, dtype=dtype, copy=True)                      # temporary array for readout
        self.data = array.reshape(-1)                                       # static 1D storage
        self.shape = tuple(array.shape)                                     # matrix shape
        self.stride = tuple(s // array.itemsize for s in array.strides)     # element wise stride from bit wise stride
        self.offset = 0                                                     # assume 0 offset
        self.dtype = dtype                                                  # store data type
        self.requires_grad = requires_grad                                  # trainable parameters or not
        self.grad = None                                                    # tensor that stores accumulated gradients
        self.grad_fn = None                                                 # backward function or None for leaf tensor