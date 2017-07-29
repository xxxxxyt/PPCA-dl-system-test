from . import autodiff

def softmax(node):
    exp_node = autodiff.exp_op(node)
    return exp_node / autodiff.reducesum_op(exp_node, 
        reduction_indices = 1, keepdims = True)
        
def softmax_cross_entropy_with_logits(logits, labels):
    h = softmax(logits)
    y = labels
    return -autodiff.reducesum_op(y * autodiff.log_op(h), reduction_indices = 1)
        
def relu(node):
    return autodiff.relu_op(node)