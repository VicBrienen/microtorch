import numpy as np

class Operation:
    def __init__(self, *parents, **attributes):
        self.parents = parents                      # reference to input tensors
        self.forward_cache = ()                     # cached arrays for backward computation
        self.attributes = attributes                # static operation attributes (e.g. axis, shape)

    def cache_for_backward(self, *xs):
        self.forward_cache = xs