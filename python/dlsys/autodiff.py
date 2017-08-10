from __future__ import absolute_import

import numpy as np
from copy import deepcopy
from ._base import lib
from ._base import cast_to_ndarray
from math import ceil
from ctypes import *

use_cpp = True

class Node(object):
    def __init__(self):
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
    def __rsub__(self, other):
        return other + (-self)
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
    def eval(self, feed_dict = None):
        from .session import Session
        sess = Session()
        return sess.run(self, feed_dict)
    def run(self, feed_dict):
        return self.eval(feed_dict)
        
        
class Op(object):
    """Op represents operations performed on nodes."""
    def __call__(self):
        new_node = Node()
        new_node.op = self
        return new_node
    def compute(self, node, input_vals, output_val, use_numpy = True):
        raise NotImplementedError
    def gradient(self, node, output_grad):
        raise NotImplementedError
    def infer_shape(self, node, input_shapes):
        raise NotImplementedError

        
class PlaceholderOp(Op):
    def __call__(self):
        new_node = Op.__call__(self)
        return new_node
    def compute(self, node, input_vals, output_val, use_numpy = True):
        assert False, "placeholder values provided by feed_dict"
    def gradient(self, node, output_grad):
        return None
    def infer_shape(self, node, input_shapes):
        assert False, "placeholder shape provided by feed_shape"

        
class VariableOp(Op):
    def __call__(self):
        new_node = Op.__call__(self)
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
        const_val = cast_to_ndarray(const_val)
        new_node.const_attr = const_val
        return new_node
    def compute(self, node, input_vals, output_val, use_numpy = True):
        assert len(input_vals) == 0
        assert node.const_attr is not None
        output_val[:] = node.const_attr
            
    def gradient(self, node, output_grad):
        return None
    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 0
        assert node.const_attr is not None
        return node.const_attr.shape
    
        
class InitOp(Op):
    def __call__(self, input_nodes):
        new_node = Op.__call__(self)
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
        
        
class ReshapeOp(Op):
    def __call__(self, node, shape):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        new_node.to_shape = shape
        return new_node
    def compute(self, node, input_vals, output_val, use_numpy = True):
        assert len(input_vals) == 1
        output_val[:] = input_vals[0].reshape(node.to_shape)
    def gradient(self, node, output_grad):
        return [reshapeto_op(output_grad, node.inputs[0])]
    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        t = 1
        for i in range(len(input_shapes[0])):
            t *= input_shapes[0][i]
        output_shape = deepcopy(node.to_shape)
        for i in range(len(output_shape)):
            if output_shape[i] != -1:
                assert t % output_shape[i] == 0
                t /= output_shape[i]
        for i in range(len(output_shape)):
            if output_shape[i] == -1:
                output_shape[i] = t
        return output_shape
        
        
class ReshapeToOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        return new_node
    def compute(self, node, input_vals, output_val, use_numpy = True):
        assert len(input_vals) == 2
        output_val[:] = input_vals[0].reshape(input_vals[1].shape)
    def gradient(self, node, output_grad):
        return [reshapeto_op(output_grad, node.inputs[0])]
    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        return input_shapes[1]
        

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
        return new_node
    def compute(self, node, input_vals, output_val, use_numpy = True):
        assert len(input_vals) == 1
        node.assign_to.value = input_vals[0]
        output_val[:] = input_vals[0]
    def gradient(self, node, output_grad):
        assert False, "no gradient for assign node"
    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]
        
        
class EqualOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
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

        
class PowerOp(Op):
    def __call__(self, node_A, node_B):
        if not isinstance(node_A, Node):
            node_A = constant_op(node_A)
        if not isinstance(node_B, Node):
            node_B = constant_op(node_B)
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        return new_node
    def compute(self, node, input_vals, output_val, use_numpy = True):
        assert len(input_vals) == 2
        output_val[:] = input_vals[0] ** input_vals[1]
    def gradient(self, node, output_grad):
        assert False
        return [node.inputs[1] * node / node.inputs[0],
                log_op(node.inputs[0]) * node]
    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        assert input_shapes[1] == (1,)
        return input_shapes[0]
        
        
class ExpOp(Op):
    def __call__(self, node):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
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
        return new_node
    def compute(self, node, input_vals, output_val, use_numpy = True):
        assert len(input_vals) == 1
        output_val[:] = np.log(input_vals[0])
    def gradient(self, node, output_grad):
        return [output_grad / node.inputs[0]]
    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]
        
        
class NegOp(Op):
    def __call__(self, node):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
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
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy = True):
        assert len(input_vals) == 2
        output_val[:] = input_vals[0] + input_vals[1]

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
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy = True):
        assert len(input_vals) == 2
        output_val[:] = input_vals[0] * input_vals[1]
                
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
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy = True):
        if use_cpp:
            A = input_vals[0].astype(np.float32)
            B = input_vals[1].astype(np.float32)
            C = np.zeros(output_val.shape).astype(np.float32)
            A_data = A.ctypes.data_as(POINTER(c_float))
            B_data = B.ctypes.data_as(POINTER(c_float))
            C_data = C.ctypes.data_as(POINTER(c_float))
            m = C.shape[0]
            n = C.shape[1]
            if node.matmul_attr_trans_A:
                k = A.shape[0]
            else:
                k = A.shape[1]
            lib.matmul(A_data, B_data, C_data,
                       node.matmul_attr_trans_A,
                       node.matmul_attr_trans_B,
                       m, k, n)
            output_val[:] = C
        else:
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
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy = True):
        assert len(input_vals) == 1
        output_val[:] = 0

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
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy = True):
        assert len(input_vals) == 1
        output_val[:] = 1

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
        new_node.reduction_indices = reduction_indices
        new_node.keepdims = keepdims
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy = True):
        assert len(input_vals) == 1
        assert(isinstance(input_vals[0], np.ndarray))
        output_val[:] = np.sum(input_vals[0], 
            axis = node.reduction_indices, keepdims = node.keepdims)

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
        return new_node
    def compute(self, node, input_vals, output_val, use_numpy = True):
        assert len(input_vals) == 2
        tmp = input_vals[0]
        input_shape = input_vals[0].shape
        output_shape = input_vals[1].shape
        for i in range(len(output_shape)):
            while(i < len(tmp.shape) and 
                tmp.shape[i] != output_shape[i]):
                tmp = np.sum(tmp, axis = i)
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
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy = True):
        assert len(input_vals) == 2
        tmp = input_vals[0]
        input_shape = ()
        output_shape = input_vals[1].shape
        j = 0
        for i in range(len(output_shape)):
            if j < len(tmp.shape) and output_shape[i] == tmp.shape[j]:
                input_shape = input_shape + (tmp.shape[j],)
                j = j + 1
            else:
                input_shape = input_shape + (1,)
        tmp = tmp.reshape(input_shape)
        output_val[:] = np.broadcast_to(tmp, output_shape)

    def gradient(self, node, output_grad):
        grad_A = reducesumto_op(output_grad, node.inputs[0])
        grad_B = zeroslike_op(node.inputs[1])
        return [grad_A, grad_B]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        return input_shapes[1]


class ReluOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy = True):
        assert len(input_vals) == 1
        output_val[:] = np.maximum(input_vals[0], 0)

    def gradient(self, node, output_grad):
        return [relu_gradient_op(node.inputs[0], output_grad)]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


class ReluGradientOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy = True):
        assert len(input_vals) == 2
        # heaviside function, 0.5 at x=0
        output_val[:] = (np.sign(input_vals[0]) + 1) * 0.5 * input_vals[1]

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        assert input_shapes[0] == input_shapes[1]
        return input_shapes[0]
        
        
class Conv2dOp(Op):
    def __call__(self, input, filter, strides, padding):
        new_node = Op.__call__(self)
        new_node.inputs = [input, filter]
        new_node.strides = strides
        new_node.padding = padding
        return new_node
        
    def compute(self, node, input_vals, output_val, use_numpy = True):
        assert len(input_vals) == 2
        X = input_vals[0]
        W = input_vals[1]
        batch, in_h, in_w, in_ch = X.shape
        fil_h, fil_w, in_ch, ou_ch = W.shape
        batch, ou_h, ou_w, ou_ch = output_val.shape
        strides = node.strides
            
        if node.padding == "SAME":
            pad_h = (ou_h - 1) * strides[1] + fil_h - in_h
            pad_w = (ou_w - 1) * strides[2] + fil_w - in_w
            pad_t = node.pad_t = pad_h // 2
            pad_b = node.pad_b = pad_h - pad_t
            pad_l = node.pad_l = pad_w // 2
            pad_r = node.pad_r = pad_w - pad_l
            X = np.pad(X, ((0, 0), (pad_t, pad_b), (pad_l, pad_r), \
                           (0, 0)), "constant")
            node.X_pad = X
            _, in_h, in_w, _ = X.shape
        
        if not use_cpp:
            W_col = W.reshape((fil_h * fil_w * in_ch, ou_ch))
            subX_col = np.zeros((ou_h * ou_w, fil_h * fil_w * in_ch))
            for b in range(batch):
                p = 0
                for i in range(ou_h):
                    q = 0
                    for j in range(ou_w):
                        subX = X[b, p : p + fil_h, q : q + fil_w, :]
                        subX_col[i * ou_w + j, :] = \
                            subX.reshape((1, fil_h * fil_w * in_ch))
                        q += strides[2]
                    p += strides[1]
                subY_col = np.matmul(subX_col, W_col)
                output_val[b, :] = subY_col.reshape((ou_h, ou_w, ou_ch))
        else:
            #print("python: compute conv2d")
            Y = np.zeros(output_val.shape).astype(np.float32)
            X = X.astype(np.float32)
            W = W.astype(np.float32)
            Y_data = Y.ctypes.data_as(POINTER(c_float))
            X_data = X.ctypes.data_as(POINTER(c_float))
            W_data = W.ctypes.data_as(POINTER(c_float))
            #print("python: type conversion")
            lib.conv2d(X_data, W_data, Y_data,
                       batch,
                       fil_h, fil_w,
                       in_h, in_w, in_ch,
                       ou_h, ou_w, ou_ch,
                       strides[1], strides[2])
            #print("python: compute conv2d done")
            output_val[:] = Y
        
    def gradient(self, node, output_grad):
        return [conv2dgradientx_op(node.inputs[0], \
                                   node.inputs[1], output_grad, node),
                conv2dgradientw_op(node.inputs[0], \
                                   node.inputs[1], output_grad, node)]
        
    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        batch, in_h, in_w, in_ch = input_shapes[0]
        fil_h, fil_w, in_ch, ou_ch = input_shapes[1]
        strides = node.strides
        if node.padding == "VALID":
            ou_h = ceil(1.00 * (in_h - fil_h + 1) / strides[1])
            ou_w = ceil(1.00 * (in_w - fil_w + 1) / strides[2])
        else:
            ou_h = ceil(1.00 * in_h / strides[1])
            ou_w = ceil(1.00 * in_w / strides[2])
        return (batch, int(ou_h), int(ou_w), ou_ch)
        
        
class Conv2dGradientXOp(Op):
    def __call__(self, input, filter, grad, origin):
        new_node = Op.__call__(self)
        new_node.inputs = [input, filter, grad]
        new_node.origin = origin
        return new_node
        
    def compute(self, node, input_vals, output_val, use_numpy = True):
        assert len(input_vals) == 3
        X = input_vals[0]
        W = input_vals[1]
        D = input_vals[2]
        batch, in_h, in_w, in_ch = X.shape
        fil_h, fil_w, in_ch, ou_ch = W.shape
        batch, ou_h, ou_w, ou_ch = D.shape
        strides = node.origin.strides
        
        if node.origin.padding == "SAME":
            """
            X = np.pad(X, ((0, 0), (node.origin.pad_t, node.origin.pad_b), \
                                   (node.origin.pad_l, node.origin.pad_r), \
                           (0, 0)), "constant")
            """
            X = node.origin.X_pad
            _, in_h, in_w, _ = X.shape
        
        if not use_cpp:
            W_col = W.reshape((fil_h * fil_w * in_ch, ou_ch))
            D_col = D.reshape((batch * ou_h * ou_w, ou_ch))
            DX = np.zeros((batch, in_h, in_w, in_ch))
            
            for b in range(batch):
                p = 0
                subD_col = D_col[b * ou_h * ou_w : (b + 1) * ou_h * ou_w, :]
                subDX_col = np.matmul(subD_col, W_col.T)
                for i in range(ou_h):
                    q = 0
                    for j in range(ou_w):
                        DX[b, p : p + fil_h, q : q + fil_w, :] += \
                            subDX_col[i * ou_w + j, :].reshape((fil_h, fil_w, in_ch))
                        q += strides[2]
                    p += strides[1]
        else:
            #print("python: compute conv2d_gradient_x")
            DX = np.zeros((batch, in_h, in_w, in_ch)).astype(np.float32)
            W = W.astype(np.float32)
            D = D.astype(np.float32)
            DX_data = DX.ctypes.data_as(POINTER(c_float))        
            W_data = W.ctypes.data_as(POINTER(c_float))
            D_data = D.ctypes.data_as(POINTER(c_float))
            #print("python: type")
            lib.conv2d_gradient_x(D_data, W_data, DX_data,
                                  batch,
                                  fil_h, fil_w,
                                  in_h, in_w, in_ch,
                                  ou_h, ou_w, ou_ch,
                                  strides[1], strides[2])
            #print("python: compute conv2d_gradient_x done")
        
        if node.origin.padding == "SAME":
            DX = DX[:, node.origin.pad_t : in_h - node.origin.pad_b, \
                       node.origin.pad_l : in_w - node.origin.pad_r, :]
        output_val[:] = DX
        
    def gradient(self, node, output_grad):
        assert False
        
    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 3
        return input_shapes[0]
        
        
class Conv2dGradientWOp(Op):
    def __call__(self, input, filter, grad, origin):
        new_node = Op.__call__(self)
        new_node.inputs = [input, filter, grad]
        new_node.origin = origin
        return new_node
        
    def compute(self, node, input_vals, output_val, use_numpy = True):
        assert len(input_vals) == 3
        X = input_vals[0]
        W = input_vals[1]
        D = input_vals[2]
        batch, in_h, in_w, in_ch = X.shape
        fil_h, fil_w, in_ch, ou_ch = W.shape
        batch, ou_h, ou_w, ou_ch = D.shape
        strides = node.origin.strides
        
        if node.origin.padding == "SAME":
            """
            X = np.pad(X, ((0, 0), (node.origin.pad_t, node.origin.pad_b), \
                                   (node.origin.pad_l, node.origin.pad_r), \
                           (0, 0)), "constant")
            """
            X = node.origin.X_pad
            _, in_h, in_w, _ = X.shape
            
        if not use_cpp:
            DW_col = np.zeros((fil_h * fil_w * in_ch, ou_ch))
            subX_col = np.zeros((ou_h * ou_w, fil_h * fil_w * in_ch))
            for b in range(batch):
                subD_col = D[b].reshape((ou_h * ou_w, ou_ch))
                p = 0
                for i in range(ou_h):
                    q = 0
                    for j in range(ou_w):
                        subX = X[b, p : p + fil_h, q : q + fil_w, :]
                        subX_col[i * ou_w + j, :] = \
                            subX.reshape((fil_h * fil_w * in_ch))
                        q += strides[2]
                    p += strides[1]
                DW_col[:] += np.matmul(subX_col.T, subD_col)
            output_val[:] = DW_col.reshape((fil_h, fil_w, in_ch, ou_ch))
        else:
            #print("python: compute conv2d_gradient_w")
            DW = np.zeros(output_val.shape).astype(np.float32)
            X = X.astype(np.float32)
            D = D.astype(np.float32)
            DW_data = DW.ctypes.data_as(POINTER(c_float))
            X_data = X.ctypes.data_as(POINTER(c_float))
            D_data = D.ctypes.data_as(POINTER(c_float))
            #print("python: type")
            lib.conv2d_gradient_w(D_data, X_data, DW_data,
                                  batch,
                                  fil_h, fil_w,
                                  in_h, in_w, in_ch,
                                  ou_h, ou_w, ou_ch,
                                  strides[1], strides[2])
            output_val[:] = DW
            #print("python: compute conv2d_gradient_w done")
        
    def gradient(self, node, output_grad):
        assert False
        
    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 3
        return input_shapes[1]
    
        
class MaxPoolOp(Op):
    def __call__(self, value, ksize, strides, padding):
        new_node = Op.__call__(self)
        new_node.inputs = [value]
        new_node.ksize = ksize
        new_node.strides = strides
        new_node.padding = padding
        return new_node
        
    def compute(self, node, input_vals, output_val, use_numpy = True):
        assert len(input_vals) == 1
        X = input_vals[0]
        batch, in_h, in_w, channels = X.shape
        batch, ou_h, ou_w, channels = output_val.shape
        ksize = node.ksize
        strides = node.strides
            
        if node.padding == "SAME":
            pad_h = (ou_h - 1) * strides[1] + ksize[1] - in_h
            pad_w = (ou_w - 1) * strides[2] + ksize[2] - in_w
            pad_t = node.pad_t = pad_h // 2
            pad_b = node.pad_b = pad_h - pad_t
            pad_l = node.pad_l = pad_w // 2
            pad_r = node.pad_r = pad_w - pad_l
            X = np.pad(X, ((0, 0), (pad_t, pad_b), (pad_l, pad_r), \
                           (0, 0)), "constant")
            node.X_pad = X
            _, in_h, in_w, _ = X.shape
            
        output_val[:] = 0
        p = 0
        for i in range(ou_h):
            q = 0
            for j in range(ou_w):
                subX = X[:, p : p + ksize[1], q : q + ksize[2], :]
                output_val[:, i, j, :] = np.amax(subX, axis = (1, 2))
                q += strides[2]
            p += strides[1]
            
    def gradient(self, node, output_grad):
        return [maxpoolgradient_op(node.inputs[0], output_grad, node)]
        
    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        batch, in_h, in_w, channels = input_shapes[0]
        ksize = node.ksize
        strides = node.strides
        if node.padding == "VALID":
            ou_h = ceil(1.00 * (in_h - ksize[1] + 1) / strides[1])
            ou_w = ceil(1.00 * (in_w - ksize[2] + 1) / strides[2])
        else:
            ou_h = ceil(1.00 * in_h / strides[1])
            ou_w = ceil(1.00 * in_w / strides[2])
        return (batch, int(ou_h), int(ou_w), channels)
        
        
class MaxPoolGradinetOp(Op):
    def __call__(self, node_A, node_B, origin):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.origin = origin
        return new_node
    
    def compute(self, node, input_vals, output_val, use_numpy = True):
        assert len(input_vals) == 2
        X = input_vals[0]
        D = input_vals[1]
        batch, in_h, in_w, channels = X.shape
        batch, ou_h, ou_w, channels = D.shape
        ksize = node.origin.ksize
        strides = node.origin.strides
        
        if node.origin.padding == "SAME":
            """
            X = np.pad(X, ((0, 0), (node.origin.pad_t, node.origin.pad_b), \
                                   (node.origin.pad_l, node.origin.pad_r), \
                           (0, 0)), "constant")
            """
            X = node.origin.X_pad
            _, in_h, in_w, _ = X.shape
            
        output_val[:] = 0
        p = 0
        for i in range(ou_h):
            q = 0
            for j in range(ou_w):
                subX = X[:, p : p + ksize[1], q : q + ksize[2], :]
                subo = output_val[:, p : p + ksize[1], q : q + ksize[2], :]
                subo[:] += np.equal(subX, np.max(subX, axis = (1, 2), keepdims = True)) * \
                           D[:, i : i + 1, j : j + 1, :]
                q += strides[2]
            p += strides[1]
    
    def gradient(self, node, output_grad):
        assert False
    
    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        return input_shapes[0]
        
        
class DropoutOp(Op):
    def __call__(self, input, keep_prob):
        new_node = Op.__call__(self)
        new_node.inputs = [input, keep_prob]
        return new_node
    
    def compute(self, node, input_vals, output_val, use_numpy = True):
        assert len(input_vals) == 2
        arr = np.random.random(output_val.shape)
        node.keep_status = arr <= input_vals[1]
        output_val[:] = input_vals[0] * node.keep_status
        
    def gradient(self, node, output_grad):
        return [dropoutgradient_op(output_grad, node), 0]
        
    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        return input_shapes[0]
        
        
class DropoutGradientOp(Op):
    def __call__(self, grad, origin):
        new_node = Op.__call__(self)
        new_node.inputs = [grad]
        new_node.origin = origin
        return new_node
        
    def compute(self, node, input_vals, output_val, use_numpy = True):
        assert len(input_vals) == 1
        output_val[:] = input_vals[0] * node.origin.keep_status
        
    def gradient(self, node, output_grad):
        assert False
        
    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


# Create global singletons of operators.
placeholder_op = PlaceholderOp()
variable_op = VariableOp()
constant_op = ConstantOp()

init_op = InitOp()
shape_op = ShapeOp()
reshape_op = ReshapeOp()
reshapeto_op = ReshapeToOp()
assign_op = AssignOp()
equal_op = EqualOp()
argmax_op = ArgmaxOp()

power_op = PowerOp()
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
relu_op = ReluOp()
relu_gradient_op = ReluGradientOp()
conv2d_op = Conv2dOp()
conv2dgradientx_op = Conv2dGradientXOp()
conv2dgradientw_op = Conv2dGradientWOp()
maxpool_op = MaxPoolOp()
maxpoolgradient_op = MaxPoolGradinetOp()
dropout_op = DropoutOp()
dropoutgradient_op = DropoutGradientOp()

    
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
                self.node_to_shape_map[node] = \
                    node.op.infer_shape(node, infer_shapes)
    
    def memory_plan(self, feed_shapes):
        self.infer_shape(feed_shapes)
        self.node_to_arr_map = {}
        for node in self.topo_order:
            self.node_to_arr_map[node] = np.empty(self.node_to_shape_map[node])

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
            self.memory_plan(feed_shapes)

        # Traverse graph in topo order and compute values for all nodes.
        for node in self.topo_order:
            if node in node_to_val_map:
                # Skip placeholder nodes. Values already provided by feed_dict.
                continue
            input_vals = [node_to_val_map[n] for n in node.inputs]
            node_val = self.node_to_arr_map[node]
            # node_val is modified in-place whether np.ndarray or NDArray
            node.op.compute(node, input_vals, node_val, use_numpy)
            node_to_val_map[node] = node_val

        # Collect node values.
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