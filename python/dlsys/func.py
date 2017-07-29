from ._base import *
from . import autodiff
import numpy as np

# global list of all variable initializers
_all_variable_inits = []

def placeholder(dtype = float32, shape = None, name = ""):
    v = autodiff.placeholder_op(name = name)
    return v

def Variable(init = None, dtype = float32, name = ""):
    v = autodiff.variable_op(name = name)
    if init is not None:
        if not isinstance(init, np.ndarray):
            if not isinstance(init, list):
                init = [init]
            init = np.array(init)
        c = autodiff.constant_op(init)
        _all_variable_inits.append(autodiff.assign_op(v, c))
    return v
    
def log(input):
    return autodiff.log_op(input)

def matmul(matA, matB):
    return autodiff.matmul_op(matA, matB)
    
def reduce_sum(input, reduction_indices = None):
    if not isinstance(reduction_indices, list):
        reduction_indices = [0]
    assert len(reduction_indices) == 1
    return autodiff.reducesum_op(input, reduction_indices[0])
    
def reduce_mean(input, reduction_indices = None):
    return reduce_sum(input, reduction_indices) / autodiff.shape_op(input, 
        reduction_indices)
    
def zeros(shape):
    return np.zeros(shape)
    
def equal(node_A, node_B):
    return autodiff.equal_op(node_A, node_B)
    
def argmax(node, axis = 0):
    return autodiff.argmax_op(node, axis)
    
def cast(node, dtype = float32):
    return node
    
def assign(assign_to, value):
    return autodiff.assign_op(assign_to, value)
    
def initialize_all_variables():
    global _all_variable_inits
    init_node = autodiff.init_op(_all_variable_inits)
    _all_variable_inits = []
    return init_node
    
def global_variables_initializer():
    return initialize_all_variables()
    
def gradients(output_node, node_list):
    assert isinstance(output_node, autodiff.Node)
    assert isinstance(node_list, list)
    return autodiff.gradients(output_node, node_list)
    
def random_normal(shape, mean = 0.0, stddev = 1.0):
    return np.random.normal(loc = mean, scale = stddev, size = shape)
