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
    
def conv2d(input, filter, strides, padding):
    assert isinstance(strides, list) and len(strides) == 4
    assert strides[0] == 1 and strides[3] == 1
    assert padding == "SAME" or padding == "VALID"
    return autodiff.conv2d_op(input, filter, strides, padding)
    
def max_pool(value, ksize, strides, padding):
    assert isinstance(ksize, list) and len(ksize) == 4
    assert isinstance(strides, list) and len(strides) == 4
    assert ksize[0] == 1 and ksize[3] == 1
    assert strides[0] == 1 and strides[3] == 1
    assert padding == "SAME" or padding == "VALID"
    return autodiff.maxpool_op(value, ksize, strides, padding)
    