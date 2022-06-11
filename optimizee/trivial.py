import torch
from optimizee import Optimizee
from torch.nn import MSELoss

class SimpleConvexModel(Optimizee):
    def __init__(self, dim, r=1):
        super(SimpleConvexModel, self).__init__()

        self.dim = dim
        self.r = r

        self.loss_func = MSELoss(reduction='mean')

        self.weight = 2 * torch.rand(self.dim, requires_grad=True) - 1 # Uniform sampling on (-1, 1)
        self.v = 2 * torch.rand(self.dim, requires_grad=False) - 1 # Uniform sampling on (-1, 1)

    def reset(self):
        self.weight = 2 * torch.rand(self.dim, requires_grad=True) - 1 # Uniform sampling on (-1, 1)
        self.v = 2 * torch.rand(self.dim, requires_grad=False) - 1 # Uniform sampling on (-1, 1)


    def loss(self):
        return self.loss_func(self.weight, self.v)