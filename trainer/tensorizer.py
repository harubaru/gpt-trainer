from log import logger
from typing import Dict, List, Tuple
from mmappickle import mmapdict
from torch import nn
import numpy as np
import torch
import time
import psutil
import os

def read_tensor(item: Dict) -> np.ndarray:
    dtype = item.dtype
    shape = item.shape
    buffer = memoryview(item)
    arr = np.ndarray.__new__(
        np.memmap,
        dtype=dtype,
        shape=shape,
        buffer=buffer,
        offset=0
    )
    return arr

def extract_tensors(m: torch.nn.Module) -> Tuple[torch.nn.Module, List[Dict]]:
    """
    Remove the tensors from a PyTorch model, convert them to NumPy
    arrays, and return the stripped model and tensors.
    """
    tensors = []
    for _, module in m.named_modules():
        # Store the tensors in Python dictionaries
        params = {
            name: torch.clone(param).detach().cpu().numpy()
            for name, param in module.named_parameters(recurse=False)
        }
        buffers = {
            name: torch.clone(buf).detach().cpu().numpy()
            for name, buf in module.named_buffers(recurse=False)
        }
        tensors.append({"params": params, "buffers": buffers})
    
    # Make a copy of the original model and strip all tensors and
    # buffers out of the copy.
    for _, module in m.named_modules():
        for name in ([name for name, _ in module.named_parameters(recurse=False)]
                     + [name for name, _ in module.named_buffers(recurse=False)]):
            setattr(module, name, None)   

    # Make sure the copy is configured for inference.
    m.train(False)
    return m, tensors

def replace_tensors(m: torch.nn.Module, tensors: List[Dict], device: torch.device) -> None:
    """
    Restore the tensors that extract_tensors() stripped out of a 
    PyTorch model.
    :param no_parameters_objects: Skip wrapping tensors in 
     ``torch.nn.Parameters`` objects (~20% speedup, may impact
     some models)
    """
    modules = [module for _, module in m.named_modules()] 
    for module, tensor_dict in zip(modules, tensors):
        # There are separate APIs to set parameters and buffers.
        for name, array in tensor_dict["params"].items():
            module.register_parameter(name, torch.nn.Parameter(torch.as_tensor(read_tensor(array), device=device)))
        for name, array in tensor_dict["buffers"].items():
            module.register_buffer(name, torch.as_tensor(read_tensor(array), device=device))

def tensorize(m: torch.nn.Module, path: str) -> None:
    logger.info(f'Tensorizing to {path}')
    model_map = mmapdict(path+'.model')
    b = time.time()
    m_copy, m_tensors = extract_tensors(m)
    logger.info(f'Model tensors and skeleton extracted in {(time.time()-b):.2f}s, {(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3):.2f}gb CPU RAM used')

    model_map['skeleton'] = m_copy
    model_map['tensors'] = m_tensors

def untensorize(path: str, device: torch.device) -> torch.nn.Module:
    model_map = mmapdict(path+'.model')

    logger.info(f'Loading {path}')

    b = time.time()
    m = model_map['skeleton'].to(device)
    logger.info(f'Model object skeleton loaded in {(time.time()-b):.2f}s')

    b = time.time()
    t = model_map['tensors']
    replace_tensors(m, t, device)
    logger.info(f'Model tensors loaded in {(time.time()-b):.2f}s, {(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3):.2f}gb CPU RAM used')

    return m
