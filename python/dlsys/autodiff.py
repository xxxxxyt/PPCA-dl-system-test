from __future__ import absolute_import

import copy
import numpy as np
from . import ndarray, gpu_op

class Node(object):
    def __init__(self):
        self.name = ""
        self.inputs = []
        self.op = None
    def __neg__(self):
        new_node = neg_op(self)
        return new_node
    def inv(self):
        new_node = inv_op(self)
        return new_node
    def __add__(self, other):
        if not isinstance(other, Node):
            other = constant_op(other)
        new_node = add_op(self, other)
        return new_node
    def __sub__(self, other):
        return self + (-other)
    def __mul__(self, other):
        if not isinstance(other, Node):
            other = constant_op(other)
        new_node = mul_op(self, other)
        return new_node
    def __div__(self, other):
        if not isinstance(other, Node):
            other = constant_op(other)
        return self * other.inv()
    def __rdiv__(self, other):
        if not isinstance(other, Node):
            other = constant_op(other)
        return other * self.inv()
    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = __sub__
    def __str__(self):
        return self.name


class Op(object):
    """Op represents operations performed on nodes."""
    def __call__(self):
        new_node = Node()
        new_node.op = self
        return new_node
    def compute(self, node, input_vals, output_val, use_numpy=True):
        raise NotImplementedError
    def gradient(self, node, output_grad):
        raise NotImplementedError
    def infer_shape(self, node, input_shapes):
        raise NotImplementedError

        
class PlaceholderOp(Op):
    def __call__(self, name = ""):
        new_node = Op.__call__(self)
        new_node.name = name
        return new_node
    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert False, "placeholder %s values provided by feed_dict" % node.name
    def gradient(self, node, output_grad):
        return None
    def infer_shape(self, node, input_shapes):
        assert False, "placeholder %s shape provided by feed_shape" % node.name

        
class VariableOp(Op):
    def __call__(self, name = ""):
        new_node = Op.__call__(self)
        new_node.name = name
        # wait value from assign node
        new_node.value = None
        return new_node
    def compute(self, node, input_vals, output_val, use_numpy = True):
        assert len(input_vals) == 0
        assert node.value is not None
        output_val[:] = node.value
    def gradient(self, node, output_grad):
        return None
    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 0
        assert node.value is not None
        return node.value.shape
        
        
class ConstantOp(Op):
    def __call__(self, const_val):
        new_node = Op.__call__(self)
        new_node.name = str(const_val)
        const_val = ndarray.cast_to_ndarray(const_val)
        new_node.const_attr = copy.deepcopy(const_val)
        return new_node
    def compute(self, node, input_vals, output_val, use_numpy = True):
        assert len(input_vals) == 0
        assert node.const_attr is not None
        if use_numpy:
            output_val[:] = node.const_attr
        else:
            const_attr = ndarray.array(node.const_attr)
            
    def gradient(self, node, output_grad):
        return None
    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 0
        assert node.const_attr is not None
        return node.const_attr.shape
    
        
class InitOp(Op):
    def __call__(self, input_nodes):
        new_node = Op.__call__(self)
        new_node.name = "(init node)"
        new_node.inputs = input_nodes
        return new_node
    def compute(self, node, input_vals, output_val, use_numpy = True):
        output_val = None
    def gradient(self, node, output_grad):
        assert False, "no gradient for init node"
    def infer_shape(self, node, input_shapes):
        return ()
        
        
class ShapeOp(Op):
    def __call__(self, node, reduction_indices = [0]):
        if not isinstance(reduction_indices, list):
            reduction_indices = [0]
        new_node = Op.__call__(self)
        new_node.name = "shape(%s)[%s]" % (node.name, str(reduction_indices))
        new_node.inputs = [node]
        new_node.reduction_indices = reduction_indices
        return new_node
    def compute(self, node, input_vals, output_val, use_numpy = True):
        assert len(input_vals) == 1
        input_shape = input_vals[0].shape
        output_val[:] = np.array([1.])
        for i in range(len(node.reduction_indices)):
            idx = node.reduction_indices[i]
            assert idx < len(input_shape)
            output_val[:] = output_val * input_shape[idx]
    def gradient(self, node, output_grad):
        return [0]
    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        return (1,)

class AssignOp(Op):
    def __call__(self, assign_node, input):
        if not isinstance(input, Node):
            input_node = constant_op(input)
        else:
            input_node = input
            
        new_node = Op.__call__(self)
        new_node.inputs = [input_node]
        # give value to variable from const node
        new_node.assign_to = assign_node
        new_node.name = "(%s=%s)" % (assign_node.name, input_node.name)
        return new_node
    def compute(self, node, input_vals, output_val, use_numpy = True):
        assert len(input_vals) == 1
        node.assign_to.value = copy.deepcopy(input_vals[0])
        output_val = None
    def gradient(self, node, output_grad):
        assert False, "no gradient for assign node"
    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        return ()
        
        
class EqualOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s==%s)" % (node_A.name, node_B.name)
        return new_node
    def compute(self, node, input_vals, output_val, use_numpy = True):
        assert len(input_vals) == 2
        assert input_vals[0].shape == input_vals[1].shape
        output_val[:] = input_vals[0] == input_vals[1]
    def gradient(self, node, output_grad):
        return [0, 0]
    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        assert input_shapes[0] == input_shapes[1]
        return input_shapes[0]
        
        
class ArgmaxOp(Op):
    def __call__(self, node, reduction_indices):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        new_node.name = "argmax(%s)[%d]" % (node.name, reduction_indices)
        new_node.reduction_indices = reduction_indices
        return new_node
    def compute(self, node, input_vals, output_val, use_numpy = True):
        assert len(input_vals) == 1
        output_val[:] = np.argmax(input_vals[0], axis = node.reduction_indices)
    def gradient(self, node, output_grad):
        return [0]
    def infer_shape(self, node, input_shapes):
        node.keepdims = False
        return reducesum_op.infer_shape(node, input_shapes)

        
class ExpOp(Op):
    def __call__(self, node):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        new_node.name = "exp(%s)" % node.name
        return new_node
    def compute(self, node, input_vals, output_val, use_numpy = True):
        assert len(input_vals) == 1
        output_val[:] = np.exp(input_vals[0])
    def gradient(self, node, output_grad):
        return [exp_op(node.inputs[0]) * output_grad]
    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]
        
class LogOp(Op):
    def __call__(self, node):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        new_node.name = "log(%s)" % node.name
        return new_node
    def compute(self, node, input_vals, output_val, use_numpy = True):
        assert len(input_vals) == 1
        output_val[:] = np.log(input_vals[0])
    def gradient(self, node, output_grad):
        return [1 / node.inputs[0] * output_grad]
    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]
        
class NegOp(Op):
    def __call__(self, node):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        new_node.name = "(-%s)" % node.name
        return new_node
    def compute(self, node, input_vals, output_val, use_numpy = True):
        assert len(input_vals) == 1
        output_val[:] = -input_vals[0]
    def gradient(self, node, output_grad):
        return [-output_grad]
    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]
        
class InvOp(Op):
    def __call__(self, node):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        new_node.name = "(1/%s)" % node.name
        return new_node
    def compute(self, node, input_vals, output_val, use_numpy = True):
        assert len(input_vals) == 1
        output_val[:] = 1 / input_vals[0]
    def gradient(self, node, output_grad):
        return [-1 * inv_op(node.inputs[0] * node.inputs[0]) * output_grad]
    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]

class AddOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s+%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        if use_numpy:
            # output_val[:] allows modify in-place
            output_val[:] = input_vals[0] + input_vals[1]
        else:
            gpu_op.matrix_elementwise_add(
                input_vals[0], input_vals[1], output_val)

    def gradient(self, node, output_grad):
        return [reducesumto_op(output_grad, node.inputs[0]), 
                reducesumto_op(output_grad, node.inputs[1])]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        return broadcast_rule(input_shapes[0], input_shapes[1])


class MulOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s*%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        
        if use_numpy:
            output_val[:] = input_vals[0] * input_vals[1]
        else:
            gpu_op.matrix_elementwise_multiply(
                input_vals[0], input_vals[1], output_val)
                
    def gradient(self, node, output_grad):
        return [reducesumto_op(node.inputs[1] * output_grad, node.inputs[0]), 
                reducesumto_op(node.inputs[0] * output_grad, node.inputs[1])]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        return broadcast_rule(input_shapes[0], input_shapes[1])


class MatMulOp(Op):
    def __call__(self, node_A, node_B, trans_A=False, trans_B=False):
        new_node = Op.__call__(self)
        new_node.matmul_attr_trans_A = trans_A
        new_node.matmul_attr_trans_B = trans_B
        new_node.inputs = [node_A, node_B]
        new_node.name = "MatMul(%s,%s,%s,%s)" % (
            node_A.name, node_B.name, str(trans_A), str(trans_B))
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        if use_numpy:
            if ((node.matmul_attr_trans_A is False) and
                    (node.matmul_attr_trans_B is False)):
                output_val[:] = np.matmul(input_vals[0], input_vals[1])
            elif ((node.matmul_attr_trans_A is True) and
                    (node.matmul_attr_trans_B is False)):
                output_val[:] = np.matmul(
                    np.transpose(input_vals[0]), input_vals[1])
            elif ((node.matmul_attr_trans_A is False) and
                    (node.matmul_attr_trans_B is True)):
                output_val[:] = np.matmul(
                    input_vals[0], np.transpose(input_vals[1]))
            elif ((node.matmul_attr_trans_A is True) and
                    (node.matmul_attr_trans_B is True)):
                output_val[:] = np.matmul(
                    np.transpose(input_vals[0]), np.transpose(input_vals[1]))
        else:
            gpu_op.matrix_multiply(
                input_vals[0], node.matmul_attr_trans_A,
                input_vals[1], node.matmul_attr_trans_B,
                output_val)

    def gradient(self, node, output_grad):
        if ((node.matmul_attr_trans_A is False) and
                (node.matmul_attr_trans_B is False)):
            # if Y=AB, then dA=dY B^T, dB=A^T dY
            lhs_grad = matmul_op(
                output_grad, node.inputs[1], trans_A=False, trans_B=True)
            rhs_grad = matmul_op(
                node.inputs[0], output_grad, trans_A=True, trans_B=False)
        elif ((node.matmul_attr_trans_A is True) and
                (node.matmul_attr_trans_B is False)):
            # if Y=A^T B, then dA=(dY B^T)^T=B dY^T, dB=A^T dY
            lhs_grad = matmul_op(
                node.inputs[1], output_grad, trans_A=False, trans_B=True)
            rhs_grad = matmul_op(
                node.inputs[0], output_grad, trans_A=True, trans_B=False)
        elif ((node.matmul_attr_trans_A is False) and
                (node.matmul_attr_trans_B is True)):
            # if Y=A B^T, then dA=dY B^T, dB=(A^T dY)^T=dY^T A
            lhs_grad = matmul_op(
                output_grad, node.inputs[1], trans_A=False, trans_B=True)
            rhs_grad = matmul_op(
                output_grad, node.inputs[0], trans_A=True, trans_B=False)
        elif ((node.matmul_attr_trans_A is True) and
                (node.matmul_attr_trans_B is True)):
            # if Y=A^T B^T, then dA=(dY B^T)^T=B dY^T, dB=(A^T dY)^T=dY^T A
            lhs_grad = matmul_op(
                node.inputs[1], output_grad, trans_A=False, trans_B=True)
            rhs_grad = matmul_op(
                output_grad, node.inputs[0], trans_A=True, trans_B=False)
        return [lhs_grad, rhs_grad]

    def infer_shape(self, node, input_shapes): # modified input_shapes
        tmp = input_shapes
        assert len(tmp) == 2
        for i in range(2):
            if tmp[i] == 1:
                tmp[i] = tmp[i] + (1,)
            assert len(tmp[i]) == 2
        x = tmp[0][0]
        y = tmp[1][1]
        if node.matmul_attr_trans_A:
            x = tmp[0][1]
        if node.matmul_attr_trans_B:
            y = tmp[1][0]
        output_shape = (x, y)
        if output_shape[1] == 1:
            output_shape = (output_shape[0],)
        return output_shape


class ZerosLikeOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Zeroslike(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        if use_numpy:
            output_val[:] = np.zeros(input_vals[0].shape)
        else:
            gpu_op.array_set(output_val, 0)

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]

    def infer_shape(self, node, input_shapes):
        """If input_shape is a vector, simpler to return (1,)"""
        """TODO: Your code here"""
        assert len(input_shapes) == 1
        output_shape = input_shapes[0]
        if (len(output_shape) == 2) and (output_shape[1] == 1):
            output_shape = (output_shape[0],)
        return output_shape


class OnesLikeOp(Op):
    def __call__(self, node_A):
        """Creates a node that represents np.ones(node_A.shape)."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Oneslike(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        if use_numpy:
            output_val[:] = np.ones(input_vals[0].shape)
        else:
            gpu_op.array_set(output_val, 1)

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]

    def infer_shape(self, node, input_shapes):
        """If input_shape is a vector, simpler to return (1,)"""
        """TODO: Your code here"""
        assert len(input_shapes) == 1
        output_shape = input_shapes[0]
        if (len(output_shape) == 2) and (output_shape[1] == 1):
            output_shape = (output_shape[0],)
        return output_shape


class ReduceSumOp(Op):
    def __call__(self, node_A, reduction_indices = 0, keepdims = False):
        assert isinstance(reduction_indices, int)
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "ReduceSum(%s)[%d]" % (node_A.name, reduction_indices)
        new_node.reduction_indices = reduction_indices
        new_node.keepdims = keepdims
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        if use_numpy:
            assert(isinstance(input_vals[0], np.ndarray))
            output_val[:] = np.sum(input_vals[0], 
                axis = node.reduction_indices, keepdims = node.keepdims)
        else:
            gpu_op.reduce_sum_axis_zero(input_vals[0], output_val)

    def gradient(self, node, output_grad):
        return [broadcastto_op(output_grad, node.inputs[0])]

    def infer_shape(self, node, input_shapes):
        """summation reduction axis = 0
        e.g. (3,4,5)->(4,5)
        for vector, simpler to do (3,)->(1,)
        """
        """TODO: Your code here"""
        assert len(input_shapes) == 1
        assert node.reduction_indices <= len(input_shapes[0])
        output_shape = ()
        for i in range(len(input_shapes[0])):
            now = (input_shapes[0][i],)
            if i == node.reduction_indices:
                if node.keepdims:
                    now = (1,)
                else:
                    now = ()
            output_shape = output_shape + now
        if len(output_shape) == 0:
            output_shape = (1,)
        return output_shape

        
class ReduceSumToOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "ReduceSumTo(%s,%s.shape)" % (node_A.name, node_B.name)
        return new_node
    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        tmp = copy.deepcopy(input_vals[0])
        input_shape = input_vals[0].shape
        output_shape = input_vals[1].shape
        for i in range(len(output_shape)):
            while(i < len(tmp.shape) and 
                tmp.shape[i] != output_shape[i]):
                tmp = copy.deepcopy(np.sum(tmp, axis = i))
        while len(tmp.shape) < len(output_shape):
            tmp = tmp.reshape(tmp.shape + (1,))
        output_val[:] = tmp
        assert output_val.shape == output_shape

    def gradient(self, node, output_grad):
        grad_A = broadcastto_op(output_grad, node.inputs[0])
        grad_B = zeroslike_op(node.inputs[1])
        return [grad_A, grad_B]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        return input_shapes[1]

class BroadcastToOp(Op):
    def __call__(self, node_A, node_B):
        """Creates a node that represents np.broadcast_to(node_A, node_B.shape).
        Only support axis=0. e.g. (3,4)->(2,3,4) to make gradient simple.
        """
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "BroadcastTo(%s,%s.shape)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        if use_numpy:
            tmp = copy.deepcopy(input_vals[0])
            input_shape = tmp.shape
            output_shape = input_vals[1].shape
            for i in range(len(output_shape) - len(input_shape)):
                input_shape = input_shape + (1,)
            tmp = tmp.reshape(input_shape)
            output_val[:] = np.broadcast_to(tmp, output_shape)
        else:
            gpu_op.broadcast_to(input_vals[0], output_val)

    def gradient(self, node, output_grad):
        grad_A = reducesumto_op(output_grad, node.inputs[0])
        grad_B = zeroslike_op(node.inputs[1])
        return [grad_A, grad_B]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        return input_shapes[1]


def softmax_func(y):
    b = y - np.max(y, axis=1, keepdims=True)
    expb = np.exp(b)
    softmax = expb / np.sum(expb, axis=1, keepdims=True)
    return softmax


class SoftmaxCrossEntropyOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "SoftmaxXEntropy(%s,%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        y = input_vals[0]
        y_ = input_vals[1]
        if use_numpy:
            softmax = softmax_func(y)
            cross_entropy = np.mean(
                -np.sum(y_ * np.log(softmax), axis=1), keepdims=True)
            output_val[:] = cross_entropy
        else:
            gpu_op.softmax_cross_entropy(y, y_, output_val)

    def gradient(self, node, output_grad):
        grad_A = (softmax_op(node.inputs[0]) + -1 * node.inputs[1])*output_grad
        grad_B = zeroslike_op(node.inputs[1])
        return [grad_A, grad_B]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        assert input_shapes[0] == input_shapes[1]
        return (1,)


class SoftmaxOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Softmax(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        if use_numpy:
            output_val[:] = softmax_func(input_vals[0])
        else:
            gpu_op.softmax(input_vals[0], output_val)

    def gradient(self, node, output_grad):
        # Do not directly use SoftmaxOp, use SoftmaxCrossEntropyOp instead.
        # Not allowing taking 2nd derivative of SoftmaxCrossEntropyOp.
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


class ReluOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Relu(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        if use_numpy:
            output_val[:] = np.maximum(input_vals[0], 0)
        else:
            gpu_op.relu(input_vals[0], output_val)

    def gradient(self, node, output_grad):
        return [relu_gradient_op(node.inputs[0], output_grad)]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


class ReluGradientOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "ReluGradient(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        if use_numpy:
            # heaviside function, 0.5 at x=0
            output_val[:] = (np.sign(input_vals[0]) + 1) * 0.5 * input_vals[1]
        else:
            gpu_op.relu_gradient(input_vals[0], input_vals[1], output_val)

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        assert input_shapes[0] == input_shapes[1]
        return input_shapes[0]


# Create global singletons of operators.
placeholder_op = PlaceholderOp()
variable_op = VariableOp()
constant_op = ConstantOp()

init_op = InitOp()
shape_op = ShapeOp()
assign_op = AssignOp()
equal_op = EqualOp()
argmax_op = ArgmaxOp()

exp_op = ExpOp()
log_op = LogOp()
neg_op = NegOp()
inv_op = InvOp()
add_op = AddOp()
mul_op = MulOp()
matmul_op = MatMulOp()

oneslike_op = OnesLikeOp()
zeroslike_op = ZerosLikeOp()

reducesum_op = ReduceSumOp()
reducesumto_op = ReduceSumToOp()
broadcastto_op = BroadcastToOp()
softmaxcrossentropy_op = SoftmaxCrossEntropyOp()
softmax_op = SoftmaxOp()
relu_op = ReluOp()
relu_gradient_op = ReluGradientOp()

    
class Executor(object):
    def __init__(self, eval_node_list, ctx=None):
        self.eval_node_list = eval_node_list
        self.ctx = ctx
        self.topo_order = find_topo_sort(self.eval_node_list)
        self.node_to_shape_map = None
        self.node_to_arr_map = None
        self.feed_shapes = None

    def infer_shape(self, feed_shapes):
        self.node_to_shape_map = {}
        for node in self.topo_order:
            get_shape = feed_shapes.get(node, False)
            if get_shape != False:
                self.node_to_shape_map[node] = get_shape
            else:
                infer_shapes = []
                for u in node.inputs:
                    infer_shapes.append(self.node_to_shape_map[u])
                self.node_to_shape_map[node] = node.op.infer_shape(node, infer_shapes)

    def memory_plan(self, feed_shapes):
        self.infer_shape(feed_shapes)
        self.node_to_arr_map = {}
        for node in self.topo_order:
            self.node_to_arr_map[node] = ndarray.empty(self.node_to_shape_map[node], ctx = self.ctx)

    def run(self, feed_dict, convert_to_numpy_ret_vals=False):
        def are_feed_shapes_equal(sa, sb):
            if (not isinstance(sa, dict)) or (not isinstance(sb, dict)):
                return False
            unmatched_item = set(sa.items()) ^ set(sb.items())
            return len(unmatched_item) == 0

        # Assume self.ctx is None implies numpy array and numpy ops.
        use_numpy = self.ctx is None
        node_to_val_map = {}
        for node, value in feed_dict.items():
            if use_numpy:
                # all values passed in feed_dict must be np.ndarray
                assert isinstance(value, np.ndarray)
                node_to_val_map[node] = value
            else:
                # convert values to ndarray.NDArray if necessary
                if isinstance(value, np.ndarray):
                    node_to_val_map[node] = ndarray.array(value, ctx=self.ctx)
                elif isinstance(value, ndarray.NDArray):
                    node_to_val_map[node] = value
                else:
                    assert False, "feed_dict value type not supported"

        # collect shapes for all placeholders
        feed_shapes = {}
        for node in node_to_val_map:
            feed_shapes[node] = node_to_val_map[node].shape

        # infer shape if feed_shapes changed since last run
        # e.g. call run() on test data after trainng
        if (not are_feed_shapes_equal(feed_shapes, self.feed_shapes)):
            self.infer_shape(feed_shapes)
            self.feed_shapes = feed_shapes
            # plan memory if using GPU
            if (not use_numpy):
                self.memory_plan(feed_shapes)

        # Traverse graph in topo order and compute values for all nodes.
        for node in self.topo_order:
            if node in node_to_val_map:
                # Skip placeholder nodes. Values already provided by feed_dict.
                continue
            input_vals = [node_to_val_map[n] for n in node.inputs]
            if use_numpy:
                node_val = np.empty(shape=self.node_to_shape_map[node])
            else:
                node_val = self.node_to_arr_map[node]
            # node_val is modified in-place whether np.ndarray or NDArray
            node.op.compute(node, input_vals, node_val, use_numpy)
            node_to_val_map[node] = node_val
        
        # Collect node values.
        if not use_numpy and convert_to_numpy_ret_vals:
            return [node_to_val_map[n].asnumpy() for n in self.eval_node_list]
        return [node_to_val_map[n] for n in self.eval_node_list]


def gradients(output_node, node_list):
    node_to_output_grads_list = {}
    node_to_output_grads_list[output_node] = [oneslike_op(output_node)]
    node_to_output_grad = {}
    # Traverse forward graph in reverse topological order
    reverse_topo_order = reversed(find_topo_sort([output_node]))
    for node in reverse_topo_order:
        output_grad = sum_node_list(node_to_output_grads_list[node])
        node_to_output_grad[node] = output_grad
        input_grads_list = node.op.gradient(node, output_grad)
        for i in range(len(node.inputs)):
            if node.inputs[i] not in node_to_output_grads_list:
                node_to_output_grads_list[node.inputs[i]] = []
            # Calculate partial adjoint for input nodes.
            node_to_output_grads_list[node.inputs[i]].append(
                input_grads_list[i])

    grad_node_list = [node_to_output_grad[node] for node in node_list]
    return grad_node_list
    
    
##################
# Helper Methods #
##################

def find_topo_sort(node_list):
    visited = set()
    topo_order = []
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order

def topo_sort_dfs(node, visited, topo_order):
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        topo_sort_dfs(n, visited, topo_order)
    topo_order.append(node)

def sum_node_list(node_list):
    from operator import add
    from functools import reduce
    return reduce(add, node_list)

def broadcast_rule(shape_a, shape_b):
    assert(isinstance(shape_a, tuple))
    assert(isinstance(shape_b, tuple))
    if len(shape_a) > len(shape_b):
        longer_shape, shorter_shape = shape_a, shape_b
    else:
        longer_shape, shorter_shape = shape_b, shape_a
    len_diff = len(longer_shape) - len(shorter_shape)
    for i in range(len_diff):
        # pad with leading 1s
        shorter_shape = (1,) + shorter_shape
    assert len(shorter_shape) == len(longer_shape)
    output_shape = list(longer_shape)
    for i in range(len(output_shape)):
        assert (shorter_shape[i] == longer_shape[i]) \
            or (shorter_shape[i] == 1) \
            or (longer_shape[i] == 1)
        output_shape[i] = max(shorter_shape[i], longer_shape[i])
    return tuple(output_shape)
