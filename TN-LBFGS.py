import numpy as np
from scipy.optimize import minimize

from loss_function import loss_function_norm, loss_function
from derivative_loss_function import derivative_loss_function

# Q_new = Q_old - (step_size) * \∇_Q derivative_loss_function(Q_old) 
# Parameter --> Tensor 'Q' , choosen one random single-site tensor for sake of maintaing the symmetries;
# overall generalized with factor of 8 as there are 8 equivalent sites to choose from in the||U - V(P,Q,R,S)||^2_F
# cost-function == loss_function_norm -- scalar
# gradient_cost-function = \∇_Q derivative_loss_function(Q_old) -- 4-legged tensor with 4 module indices
# \∇_Q derivative_loss_function(Q_old) is same shape as Tensor Q i.e 4 legs so the update step is consistent

def pack_complex_tensors(tensor_dict): #complex tensor_dict input
    parts = [] #list to hold the vector
    
    for key in sorted(tensor_dict.keys()): # order preserving
        tensor = tensor_dict[key]
        parts.append(tensor.real.flatten())
        parts.append(tensor.imag.flatten())
        
    return np.concatenate(parts) #real vector output

def unpack_complex_tensors(vector_real, key_shape): #real vector, shape input

    tensor_dict, real_part, imag_part = {}, {} ,{}

    for key, shape in key_shape.items():
        n_elements = np.prod(shape)
        real_part[key] = vector_real[:n_elements].reshape.shape
        imag_part[key] = vector_real[n_elements:].reshape.shape
    
    for key in key_shape.items():
        tensor_dict[key] = real_part[key] + 1j*imag_part[key]
    
    return tensor_dict

