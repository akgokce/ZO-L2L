# Studying Zeroth-Order Methods in a Learning to Learn Framework
### *CS-439, Optimization for Machine Learning Project by Abdulkadir Gokce, Mahammad Ismayilzada, Mert Cemri*

This repository contains the code for CS-439, OptML project done by *Abdulkadir Gokce, Mahammad Ismayilzada, Mert Cemri* and has been largely adapted from the original [repo](https://github.com/ryoungj/ZO-L2L) for [Learning to Learn by Zeroth-Order Oracle](https://openreview.net/forum?id=ryxz8CVYDH), which extends the learning to learn (L2L) framework to zeroth-order (ZO) optimization.

## Setup
We provide a `requirements.txt` file that you can use to setup all the dependencies needed to run experiments in this project.
```sh
pip install -r requirements.txt
```
Minimum python version required is Python=3.6.

## Overview
As mentioned, this repository has been forked from the original [repo](https://github.com/ryoungj/ZO-L2L) for [Learning to Learn by Zeroth-Order Oracle](https://openreview.net/forum?id=ryxz8CVYDH) and adapted for our experiments. Below we outline the repository structure and provide details on each file and modifications made by us.

## Repo structure
- [run.ipynb](run.ipynb)

    This file contains the code to reproduce figures used in our report. Due to the nature of the experiments we ran, we couldn't put all the scripts in one file to reproduce the experiments themselves, but instead we provide a [scripts](scripts) folder that contains all the testing and training scripts we used to run our experiments. When we run the experiments, all output artifacts (loss values, models etc.) are captured in the [output](output) folder from which we read in [run.ipynb](run.ipynb) to produce our final figures. For more info on these directories, please refer to their respective sections below.

- [main_attack.py](main_attack.py)

    This file is the main entrypoint in this project for almost all our experiments. It contains the code to train and test our optimizers on attack tasks. This file has only been modified to control the compute precision, to enable optimization tricks like random scaling and convexity injection and to log results to wandb.ai.

- [train_task_list.py](train_task_list)

    This file contains the configurations for our experiments. Our scripts specify the task to be run and main_attack.py reads from this file to locate the experiment configurations.

- [utils.py](utils.py)

    This file contains some utilities for our experiments such as adjusting compute precision. All code in this file has been added by us.

- [scripts](scripts)

    This directory contains various training and testing scripts to run our experiments. We have named each script such that it is largely self-explanatory (i.e. specifies attack task, optimization trick used etc.)

- [output](output)

    This directory contains various outputs of our experiments such as learned optimizers, models, loss values etc. We use these artifacts in our [run.ipynb](run.ipynb) to produce final figures for our report. Name of each artifact corresponds to the name of the experiment script in [scripts](scripts) directory.

- [notebooks](notebooks)

    This directory contains notebooks to train black-box attack models on different datasets that we then used to test our learned optimizer's generalizability. It also contains a notebook called [VarReduced_ZO_combine.ipynb](notebooks/VarReduced_ZO_combine.ipynb) which we wrote to combine our modified `updateRNN` with the static `queryRNN` model to produce the final `VarReducedZOOptimizer` used to test the generalization abilities the learned optimizers. All code in this directory has been added by us.

- [optimizee](optimizee)

    This directory contains code for our attack models with additions and modifications for training tricks. All code in [cifar.py](optimizee/cifar.py) and [trivial.py](optimizee/trivial.py) has been added by us. Code in [mnist.py] has been modified to work with our training tricks and has been extended with attack models corresponding to our generalization experiments.

- [nn_optimizer](nn_optimizer)

    This directory contains the files for various ZO optimizers used in our experiments. [__init__.py](nn_optimizer/__init__.py) and [basezoopt.py](nn_optimizer/basezoopt.py) contains code for base optimizer and standard ZO optimizers and has not been modified by us. [zoopt.py](nn_optimizer/zoopt.py) contains the code for our main focus in this project, the learned optimizer `ZOOptimizer` (only using `updateRNN`) and `VarReducedZOOptimizer` (both `updateRNN` and `queryRNN`) and its variants enhanced with our training tricks.

- [ckpt](ckpt)

    This directory contains the files for our saved attack models.

- [data](data)

    This directory contains various data files such as indices of data samples correctly classified by our attack models so that we can attack these samples in our testing experiments.

