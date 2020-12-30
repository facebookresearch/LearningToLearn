
import numpy as np
import torch
import operator
from functools import reduce


prod = lambda l: reduce(operator.mul, l, 1)
torch.set_default_tensor_type(torch.DoubleTensor)


def vector3_to_skew_symm_matrix(vec3):
    vec3 = convert_into_at_least_2d_pytorch_tensor(vec3)
    batch_size = vec3.shape[0]
    skew_symm_mat = vec3.new_zeros((batch_size, 3, 3))
    skew_symm_mat[:, 0, 1] = -vec3[:, 2]
    skew_symm_mat[:, 0, 2] = vec3[:, 1]
    skew_symm_mat[:, 1, 0] = vec3[:, 2]
    skew_symm_mat[:, 1, 2] = -vec3[:, 0]
    skew_symm_mat[:, 2, 0] = -vec3[:, 1]
    skew_symm_mat[:, 2, 1] = vec3[:, 0]
    return skew_symm_mat



def convert_into_pytorch_tensor(variable):
    if isinstance(variable, torch.Tensor):
        return variable
    elif isinstance(variable, np.ndarray):
        return torch.Tensor(variable)
    else:
        return torch.Tensor(variable)


def convert_into_at_least_2d_pytorch_tensor(variable):
    tensor_var = convert_into_pytorch_tensor(variable)
    if len(tensor_var.shape) == 1:
        return tensor_var.unsqueeze(0)
    else:
        return tensor_var

