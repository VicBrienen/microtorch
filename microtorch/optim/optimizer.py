class Optimizer:
    def __init__(self, params):
        self.params = list(params)

        for p in params:
            if p.requires_grad and p.grad_fn is not None:
                raise ValueError("Optimizer received non parameter")

    def zero_grad(self):
        for p in self.params:
            p.grad = None