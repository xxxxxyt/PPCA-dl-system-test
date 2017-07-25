import numpy as np
from . import ndarray
from . import autodiff
from ._base import *

class Session(object):
    def __enter__(self):
        return self
    def __exit__(self, e_t, e_v, t_b):
        return None
    def run(self, fetch, feed_dict = None):
        if not isinstance(fetch, list):
            fetch = [fetch]
        if not isinstance(feed_dict, dict):
            feed_dict = {}
            
        for node in feed_dict:
            value = feed_dict[node]
            if not isinstance(value, np.ndarray):
                if not isinstance(value, list):
                    value = [value]
                value = np.array(value)
            feed_dict[node] = value
            
        executor = autodiff.Executor(fetch)
        res = executor.run(feed_dict)
        return res[0]


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
    
def assign(assign_to, value):
    return autodiff.assign_op(assign_to, value)

def initialize_all_variables():
    global _all_variable_inits
    init_node = autodiff.init_op(_all_variable_inits)
    _all_variable_inits = []
    return init_node
    
def global_variables_initializer():
    return initialize_all_variables()
    
def reduce_sum(input):
    return autodiff.reducesumaxiszero_op(input)
    
def zeros(shape):
    return np.zeros(shape)

"""
def gradients(ys, xs, grad_ys=None):
    if isinstance(ys, list):
        ys = symbol.Group(ys)
    g = graph.create(ys)
    g._set_symbol_list_attr('grad_ys', ys)
    g._set_symbol_list_attr('grad_xs', xs)
    ny = len(ys.list_output_names())
    if grad_ys is None:
        grad_ys = [symbol.ones_like(ys[i]) for i in range(ny)]
    g._set_symbol_list_attr('grad_ys_out_grad', grad_ys)
    sym = g.apply('Gradient').symbol
    nx = len(xs) if isinstance(xs, list) else len(xs.list_output_names())
    ret = [sym[i] for i in range(nx)]
    return ret
"""