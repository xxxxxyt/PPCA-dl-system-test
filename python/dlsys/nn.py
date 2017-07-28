from . import autodiff

def softmax(node):
    #return autodiff.softmax_op(node)
    exp_node = autodiff.exp_op(node)
    return exp_node / autodiff.reducesum_op(exp_node, 
        reduction_indices = 1, keepdims = True)