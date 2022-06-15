import torch
from optimizee import Optimizee
from torch.nn import MSELoss, Parameter

class SimpleConvexModel(Optimizee):
    def __init__(self, dim, r=1, cuda=False, gpu_num=None, dtype=torch.float64):
        super(SimpleConvexModel, self).__init__()

        self.dim = dim
        self.r = r
        self.cuda = cuda
        self.gpu_num = gpu_num
        self.dtype = dtype

        self.loss_func = MSELoss(reduction='mean')

        # Uniform sampling on (-1, 1)
        weight = 2 * torch.rand(self.dim, requires_grad=True, dtype=self.dtype) - 1
        self.weight = Parameter(weight)
        self.v = 2 * torch.rand(self.dim, requires_grad=False, dtype=self.dtype) - 1

        if self.cuda:
            self.weight = Parameter(self.weight.cuda(self.gpu_num))
            self.v = self.v.cuda(self.gpu_num)

    def re_initialize(self):
        # Generate a new quadratic optimization problem
        weight = 2 * torch.rand(self.dim, requires_grad=True, dtype=self.dtype) - 1
        self.weight = Parameter(weight)
        self.v = 2 * torch.rand(self.dim, requires_grad=False, dtype=self.dtype) - 1

        if self.cuda:
            self.weight = Parameter(self.weight.cuda(self.gpu_num))
            self.v = self.v.cuda(self.gpu_num)

    # def reset(self):
    #     # Uniform sampling on (-1, 1)
    #     weight = 2 * torch.rand(self.dim, requires_grad=True, dtype=self.dtype) - 1
    #     self.weight = Parameter(weight)
    #     self.v = 2 * torch.rand(self.dim, requires_grad=False, dtype=self.dtype) - 1

    #     if self.cuda:
    #         self.weight = Parameter(self.weight.cuda(self.gpu_num))
    #         self.v = self.v.cuda(self.gpu_num)


    def loss(self):
        return self.loss_func(self.weight, self.v)