import torch
from typing import Union

def set_precision(x:Union[torch.tensor, torch.nn.Module], precision:Union[torch.dtype, str]) -> Union[torch.tensor, torch.nn.Module]:
    if isinstance(precision, str):
        if precision in ['full', 'float', 'float32', 'torch.float32']:
            precision = torch.float32
        elif precision in ['double', 'float64', 'torch.float64']:
            precision = torch.float64
        elif precision in ['half', 'float16', 'torch.float16']:
            precision = torch.float16
        else:
            raise NotImplementedError

    return x.type(precision)


def set_default_precision(precision:Union[torch.dtype, str]) -> None:
    if isinstance(precision, str):
        if precision in ['full', 'float', 'float32', 'torch.float32']:
            precision = torch.float32
        elif precision in ['double', 'float64', 'torch.float64']:
            precision = torch.float64
        elif precision in ['half', 'float16', 'torch.float16']:
            precision = torch.float16
        else:
            raise NotImplementedError
            
    torch.set_default_dtype(precision)


def get_precision_dtype(dtype:str) -> torch.dtype:
    if dtype in ['full', 'float', 'float32', 'torch.float32']:
        dtype = torch.float32
    elif dtype in ['double', 'float64', 'torch.float64']:
        dtype = torch.float64
    elif dtype in ['half', 'float16', 'torch.float16']:
        dtype = torch.float16
    else:
        raise NotImplementedError

    return dtype
