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
        optimizer = autodiff.init_op(assign_nodes) # not the same as initialization
        return optimizer
        
        
class AdamOptimizer:
    def __init__(self, learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999,
        epsilon = 1e-08):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
    def minimize(self, cost_function):
        assert isinstance(cost_function, autodiff.Node)
        topo_order = autodiff.find_topo_sort([cost_function])
        para = []
        for node in topo_order:
            if isinstance(node.op, autodiff.VariableOp):
                para.append(node)
        grad = gradients(node, para)
        
        t = Variable(0, "t")
        m = [Variable(0, "m") for i in range(len(para))]
        v = [Variable(0, "v") for i in range(len(para))]
        # initilize when global initialization
        
        t_ = assign(t, t + 1)
        lr_ = self.learning_rate * \
            sqrt(1 - power(self.beta2, t_)) / (1 - power(self.beta1, t_))
        assign_nodes = []
        for i in range(len(para)):
            m_ = assign(m[i], self.beta1 * m[i] + (1 - self.beta1) * grad[i])
            v_ = assign(v[i], self.beta2 * v[i] + (1 - self.beta2) * grad[i] * grad[i])
            para_ = assign(para[i], para[i] - lr_ * m_ / (sqrt(v_) + self.epsilon))
            assign_nodes.append(para_)
        optimizer = autodiff.init_op(assign_nodes)
        return optimizer
