import torch
import torch.nn as nn

class MyMixUp:
    def __init__(self, alpha=1.0, seed=0):
        self.alpha = alpha
        torch.manual_seed(seed)

    def __call__(self, x, y):
        if self.alpha > 0:
            lam = torch.distributions.Beta(self.alpha, self.alpha)
        else:
            lam = 1
        lam = lam.sample().item()
        batch_size = x.size(0)  
        index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam