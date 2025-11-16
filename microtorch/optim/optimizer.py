class Optimizer:
    def __init__(self, params, defaults):
        self.param_groups = []
        self.state = {}

        for param in params:
            self.param_groups.append({
                'params': param,
                **defaults
            })

    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                p.grad = None