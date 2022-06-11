'''The model and optimizee used when training.'''
from torch import optim

import nn_optimizer
import optimizee

# Only `MNIST attack` task support yet
tasks = {
    # train ZO optimizer (UpdateRNN only) for MNIST attack
    'ZOL2L-Attack': {
        'nn_optimizer': nn_optimizer.zoopt.ZOOptimizer,
        'optimizee': optimizee.mnist.MnistAttack,
        'batch_size': 1,
        'test_batch_size': 1,
        'lr': 1e-3,
        "max_epoch": 20,
        'optimizer_steps': 200,
        'test_optimizer_steps': 200,
        'attack_model': optimizee.mnist.MnistConvModel,
        'attack_model_ckpt': "./ckpt/attack_model/mnist_cnn.pt",
        'tests': {
            'optimizee': optimizee.mnist.MnistAttack,
            'test_indexes': list(range(1, 11)),  # test image indexes
            'test_num': 10,  # number of independent attacks
            'n_steps': 200,
            'test_batch_size': 1,
            'nn_opt': nn_optimizer.zoopt.ZOOptimizer,
            'base_opt': nn_optimizer.basezoopt.BaseZOOptimizer,
            'base_lr': 4,
            'mean': 0.1307,
            'std': 0.3081,
            "num_classes": 10
        }
    },
    # train ZO optimizer (both UpdateRNN and QueryRNN) for MNIST attack
    'VarReducedZOL2L-Attack': {
        'nn_optimizer': nn_optimizer.zoopt.VarReducedZOOptimizer,
        'optimizee': optimizee.mnist.MnistAttack,
        'batch_size': 1,
        'test_batch_size': 1,
        'lr': 0.005,
        "max_epoch": 40,
        'optimizer_steps': 200,
        'test_optimizer_steps': 200,
        'attack_model': optimizee.mnist.MnistConvModel,
        'attack_model_ckpt': "./ckpt/attack_model/mnist_cnn.pt",
        'tests': {
            'optimizee': optimizee.mnist.MnistAttack,
            'test_indexes': list(range(1, 11)),  # test image indexes
            'test_num': 10,  # number of independent attacks
            'n_steps': 200,
            'test_batch_size': 1,
            'nn_opt': nn_optimizer.zoopt.VarReducedZOOptimizer,
            'base_opt': nn_optimizer.basezoopt.BaseZOOptimizer,
            'base_lr': 4,
            'sign_opt': nn_optimizer.basezoopt.SignZOOptimizer,
            'sign_lr': 8,
            'adam_opt': nn_optimizer.basezoopt.AdamZOOptimizer,
            'adam_lr': 8,
            'adam_beta_1': 0.9,
            'adam_beta_2': 0.996,
            'mean': 0.1307,
            'std': 0.3081,
            "num_classes": 10
            # 'nn_opt_no_query': nn_optimizer.zoopt.VarReducedZOOptimizer,
            # 'nn_opt_no_update': nn_optimizer.zoopt.VarReducedZOOptimizer,
            # 'nn_opt_guided': nn_optimizer.zoopt.VarReducedZOOptimizer,
        }
    },
    # train ZO optimizer (both UpdateRNN and QueryRNN) for EMNIST attack
    'VarReducedZOL2L-Attack-EMNIST': {
        'nn_optimizer': nn_optimizer.zoopt.VarReducedZOOptimizer,
        'optimizee': optimizee.mnist.EMnistAttack,
        'batch_size': 1,
        'test_batch_size': 1,
        'lr': 0.005,
        "max_epoch": 40,
        'optimizer_steps': 200,
        'test_optimizer_steps': 200,
        'attack_model': optimizee.mnist.EMnistConvModel,
        'attack_model_ckpt': "./ckpt/attack_model/emnist_cnn.pt",
        'tests': {
            'optimizee': optimizee.mnist.EMnistAttack,
            'test_indexes': list(range(1, 3)),  # test image indexes
            'test_num': 10,  # number of independent attacks
            'n_steps': 200,
            'test_batch_size': 1,
            'nn_opt': nn_optimizer.zoopt.VarReducedZOOptimizer,
            'base_opt': nn_optimizer.basezoopt.BaseZOOptimizer,
            'base_lr': 4,
            'sign_opt': nn_optimizer.basezoopt.SignZOOptimizer,
            'sign_lr': 8,
            'adam_opt': nn_optimizer.basezoopt.AdamZOOptimizer,
            'adam_lr': 8,
            'adam_beta_1': 0.9,
            'adam_beta_2': 0.996,
            "mean": 0.1751,
            "std": 0.3332,
            "num_classes": 47
            # 'nn_opt_no_query': nn_optimizer.zoopt.VarReducedZOOptimizer,
            # 'nn_opt_no_update': nn_optimizer.zoopt.VarReducedZOOptimizer,
            # 'nn_opt_guided': nn_optimizer.zoopt.VarReducedZOOptimizer,
        }
    },
    # train ZO optimizer (both UpdateRNN and QueryRNN) for FashionMNIST attack
    'VarReducedZOL2L-Attack-FMNIST': {
        'nn_optimizer': nn_optimizer.zoopt.VarReducedZOOptimizer,
        'optimizee': optimizee.mnist.FMnistAttack,
        'batch_size': 1,
        'test_batch_size': 1,
        'lr': 0.005,
        "max_epoch": 40,
        'optimizer_steps': 200,
        'test_optimizer_steps': 200,
        'attack_model': optimizee.mnist.FMnistConvModel,
        'attack_model_ckpt': "./ckpt/attack_model/fmnist_cnn.pt",
        'tests': {
            'optimizee': optimizee.mnist.FMnistAttack,
            'test_indexes': list(range(1, 3)),  # test image indexes
            'test_num': 10,  # number of independent attacks
            'n_steps': 200,
            'test_batch_size': 1,
            'nn_opt': nn_optimizer.zoopt.VarReducedZOOptimizer,
            'base_opt': nn_optimizer.basezoopt.BaseZOOptimizer,
            'base_lr': 4,
            'sign_opt': nn_optimizer.basezoopt.SignZOOptimizer,
            'sign_lr': 8,
            'adam_opt': nn_optimizer.basezoopt.AdamZOOptimizer,
            'adam_lr': 8,
            'adam_beta_1': 0.9,
            'adam_beta_2': 0.996,
            "mean": 0.2860,
            "std": 0.3530,
            "num_classes": 10
            # 'nn_opt_no_query': nn_optimizer.zoopt.VarReducedZOOptimizer,
            # 'nn_opt_no_update': nn_optimizer.zoopt.VarReducedZOOptimizer,
            # 'nn_opt_guided': nn_optimizer.zoopt.VarReducedZOOptimizer,
        }
    },
    # train ZO optimizer (both UpdateRNN and QueryRNN) for CIFAR10 attack
    'VarReducedZOL2L-Attack-CIFAR10': {
        'nn_optimizer': nn_optimizer.zoopt.VarReducedZOOptimizer,
        'optimizee': optimizee.cifar.CIFAR10Attack,
        'batch_size': 1,
        'test_batch_size': 1,
        'lr': 0.005,
        "max_epoch": 40,
        'optimizer_steps': 200,
        'test_optimizer_steps': 200,
        'attack_model': optimizee.cifar.CIFAR10Model,
        'attack_model_ckpt': "./ckpt/attack_model/cifar_cnn.pt",
        'tests': {
            'optimizee': optimizee.cifar.CIFAR10Attack,
            'test_indexes': list(range(1, 3)),  # test image indexes
            'test_num': 10,  # number of independent attacks
            'n_steps': 200,
            'test_batch_size': 1,
            'nn_opt': nn_optimizer.zoopt.VarReducedZOOptimizer,
            'base_opt': nn_optimizer.basezoopt.BaseZOOptimizer,
            'base_lr': 4,
            'sign_opt': nn_optimizer.basezoopt.SignZOOptimizer,
            'sign_lr': 8,
            'adam_opt': nn_optimizer.basezoopt.AdamZOOptimizer,
            'adam_lr': 8,
            'adam_beta_1': 0.9,
            'adam_beta_2': 0.996,
            "mean": 0.4809,
            "std": 0.2333,
            "num_classes": 10
            # 'nn_opt_no_query': nn_optimizer.zoopt.VarReducedZOOptimizer,
            # 'nn_opt_no_update': nn_optimizer.zoopt.VarReducedZOOptimizer,
            # 'nn_opt_guided': nn_optimizer.zoopt.VarReducedZOOptimizer,
        }
    },
}
