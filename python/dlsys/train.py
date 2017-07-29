from .session import Session
from . import autodiff
from ._base import *
from .func import *

class GradientDescentOptimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    def minimize(self, cost_function):
        assert isinstance(cost_function, autodiff.Node)
        topo_order = autodiff.find_topo_sort([cost_function])
        para = []
        for node in topo_order:
            if isinstance(node.op, autodiff.VariableOp):
                para.append(node)
        grad = gradients(node, para)
        
        assign_nodes = []
        for i in range(len(para)):
            assign_nodes.append(assign(para[i], 
                para[i] - self.learning_rate * grad[i]))
        optimizer = autodiff.init_op(assign_nodes)
        return optimizer
