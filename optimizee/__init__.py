import torch
from torch import nn
import os
from functools import reduce
from operator import mul
from torch.autograd import Variable, Function

from itertools import product
import time

class Optimizee(nn.Module):
    def __init__(self):
        super(Optimizee, self).__init__()

    @staticmethod
    def dataset_loader(*input):
        raise NotImplementedError

    def loss(self, *input):
        raise NotImplementedError

    def reset(self):
        for module in self.modules():
            if len(module._parameters) != 0:
                module._parameters['weight'] = Variable(module._parameters['weight'].data)
                try:
                    module._parameters['bias'] = Variable(module._parameters['bias'].data)
                except:
                    pass

    def get_params(self):
        params = []
        for name, module in self.named_modules():
            if len(module._parameters) != 0:
                params.append(module._parameters['weight'].data.view(-1))
                try:
                    params.append(module._parameters['bias'].data.view(-1))
                except:
                    pass
        return torch.cat(params)

    def set_params(self, flat_params):
        # Restore original shapes
        offset = 0
        trial = 0
        for module in self.modules():
            if len(module._parameters) != 0:
                """
                if trial == 0:
                  print(module._parameters["weight"])
                  trial = 1
                  print("printed")
                  time.sleep(20)
                """
                weight_shape = module._parameters['weight'].size()
                weight_flat_size = reduce(mul, weight_shape, 1)
                module._parameters['weight'].data = flat_params[
                                               offset:offset + weight_flat_size].view(*weight_shape)
                try:
                    bias_shape = module._parameters['bias'].size()
                    bias_flat_size = reduce(mul, bias_shape, 1)
                    module._parameters['bias'].data = flat_params[
                                                 offset + weight_flat_size:offset + weight_flat_size + bias_flat_size].view(
                        *bias_shape)
                except:
                    bias_flat_size = 0
                offset += weight_flat_size + bias_flat_size

    def get_params_size(self):
        return self.get_params().size(0)

    def get_flat_params(self):
        params = []
        """
        for name, module in self.model.named_modules():
            if len(module._parameters) != 0:
                params.append(module._parameters['weight'].view(-1))
                try:
                    params.append(module._parameters['bias'].view(-1))
                except:
                    pass
        """
        trial = 1
        for name, module in self.named_modules():
            if len(module._parameters) != 0:
                
                if trial == 0:
                  print(module._parameters["weight"])
                  trial = 1
                  print("printed")
                  time.sleep(5)
                
                params.append(module._parameters['weight'].view(-1))
                try:
                    params.append(module._parameters['bias'].view(-1))
                except:
                    pass
        return torch.cat(params)
        
    def set_flat_params(self, flat_params):
        # Restore original shapes
        
        #print(flat_params.size())
        #time.sleep(20)
        offset = 0
        for module in self.modules():
            if len(module._parameters) != 0:
                
                weight_shape = module._parameters['weight'].size()
                weight_flat_size = reduce(mul, weight_shape, 1)
                try:
                  module._parameters['weight'] = flat_params[
                                             offset:offset + weight_flat_size].view(*weight_shape)
                except:
                  print(weight_shape,weight_flat_size)
                  raise Exception
                try:
                    bias_shape = module._parameters['bias'].size()
                    bias_flat_size = reduce(mul, bias_shape, 1)
                    module._parameters['bias'] = flat_params[
                                                 offset + weight_flat_size:offset + weight_flat_size + bias_flat_size].view(
                        *bias_shape)
                except:
                    bias_flat_size = 0
                offset += weight_flat_size + bias_flat_size
                #print(offset)

class MetaModel:
    def __init__(self, model):
        self.model = model

    def reset(self):
        for module in self.model.modules():
            if len(module._parameters) != 0:
                module._parameters['weight'] = Variable(module._parameters['weight'].data)
                try:
                    module._parameters['bias'] = Variable(module._parameters['bias'].data)
                except:
                    pass

    def get_flat_params(self):
        params = []
        """
        for name, module in self.model.named_modules():
            if len(module._parameters) != 0:
                params.append(module._parameters['weight'].view(-1))
                try:
                    params.append(module._parameters['bias'].view(-1))
                except:
                    pass
        """
        for name, module in self.model.named_modules():
            if len(module._parameters) != 0:
                params.append(module._parameters['weight'].view(-1))
                try:
                    params.append(module._parameters['bias'].view(-1))
                except:
                    pass
        return torch.cat(params)



    def set_flat_params(self, flat_params):
        # Restore original shapes
        offset = 0
        for module in self.model.modules():
            if len(module._parameters) != 0:
                weight_shape = module._parameters['weight'].size()
                #print(weight_shape)
                weight_flat_size = reduce(mul, weight_shape, 1)
                #print(weight_flat_size)
                module._parameters['weight'] = flat_params[
                                               offset:offset + weight_flat_size].view(*weight_shape)
                try:
                    bias_shape = module._parameters['bias'].size()
                    bias_flat_size = reduce(mul, bias_shape, 1)
                    module._parameters['bias'] = flat_params[
                                                 offset + weight_flat_size:offset + weight_flat_size + bias_flat_size].view(
                        *bias_shape)
                except:
                    bias_flat_size = 0
                offset += weight_flat_size + bias_flat_size

    def copy_params_from(self, model):
        for modelA, modelB in zip(self.model.parameters(), model.parameters()):
            modelA.data.copy_(modelB.data)

    def copy_params_to(self, model):
        for modelA, modelB in zip(self.model.parameters(), model.parameters()):
            modelB.data.copy_(modelA.data)


class AttackModel:
    def __init__(self, model):
        self.model = model


class CustomLoss(Function):
    """
    @staticmethod
    def forward(ctx, weight, inputs, tgt, loss_func):
        ctx.loss_func = loss_func
        ctx.save_for_backward(weight, inputs, tgt)
        return loss_func(weight, inputs, tgt)
    """
    @staticmethod
    def forward(ctx, weight, fx, tgt, loss_func):
        ctx.loss_func = loss_func
        ctx.save_for_backward(weight, fx, tgt)
        """
        print("hiiii")
        print("hiiii")
        print("hiiii")
        """
        
        mll = loss_func(weight,fx, tgt)
        
        #print(mll)
        return mll
    """
    @staticmethod
    def backward(ctx, grad_output):
        
        loss_func = ctx.loss_func
        weight, inputs, tgt = ctx.saved_tensors

        #print(weight.size())
        grad_weight = torch.zeros_like(weight)

        eps = 1e-6

        weight_shape = weight.size()
        #weight = weight.data

        orig = weight.clone()

        weight_a = orig - eps
        weight_b = orig + eps
        
        print(weight_a.size())
        
        outa = loss_func(weight_a, inputs, tgt, batch_weight=True).clone()
        outb = loss_func(weight_b, inputs, tgt, batch_weight=True).clone()
        
        print(outa.size())
        
        r = (outb - outa) / (2 * eps)
        
        print(r.size())
        
        grad_weight = r.view(*weight_shape).detach()
       
        return grad_weight * grad_output, None, None, None
               
    """
    @staticmethod
    def backward(ctx, grad_output):
        
        #print("backwaard")
        #print("backwaard")
        #print("backwaard")
        
        loss_func = ctx.loss_func
        weight, inputs, tgt = ctx.saved_tensors

        
        grad_weight = torch.zeros_like(weight)

        eps = 1e-6

        weight_shape = weight.size()
        weight = weight.data
        weight_a_array = []
        weight_b_array = []
        
        """
        for idx in product(*[range(m) for m in grad_weight.size()]):
            orig = weight[idx].item()
            weight_a = weight.clone()
            weight_a[idx] = orig - eps
            weight_a_array.append(weight_a)

            #del weight_a
            weight_b = weight.clone()
            weight_b[idx] = orig + eps
            weight_b_array.append(weight_b)

            #del weight_b
            
        weight_a_tensor = torch.stack(weight_a_array)
        weight_b_tensor = torch.stack(weight_b_array)
        
        outa = loss_func(weight_a_tensor, inputs, tgt, batch_weight=True).clone()
        outb = loss_func(weight_b_tensor, inputs, tgt, batch_weight=True).clone()
        
        r = (outb - outa) / (2 * eps)
        """
        grads = []
        
        for idx in product(*[range(m) for m in grad_weight.size()]):
            orig = weight[idx].item()
            weight_a = weight.clone()
            weight_a[idx] = orig - eps
            weight_a_array.append(weight_a)

            weight_b = weight.clone()
            weight_b[idx] = orig + eps
            weight_b_array.append(weight_b)

            outa = loss_func(weight_a, inputs, tgt, batch_weight=True).clone()
            outb = loss_func(weight_b, inputs, tgt, batch_weight=True).clone()
        
            r = (outb - outa) / (2 * eps)
            
           
            #print(r)
            grads.append(r.cuda("cuda:0"))
        grads = torch.tensor(grads)
        
        grad_weight = grads.view(*weight_shape)
        
        return grad_weight * grad_output, None, None, None
    

custom_loss = CustomLoss.apply


from . import mnist