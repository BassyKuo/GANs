#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import scipy.misc

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak*x)

def one_hot(size):
    y = np.zeros(size)
    idx = np.random.randint(size[1], size=[size[0]])
    for b, c in enumerate(idx):
        y[b,c] = 1
    return y

def batch_norm(X, eps=1e-8, g=None, b=None):
    if X.get_shape().ndims == 4:
        # mean = tf.reduce_mean(X, [0,1,2])
        # std = tf.reduce_mean( tf.square(X-mean), [0,1,2] )
        mean, std = tf.nn.moments(X, [0,1,2]) #shape=(channel, )
        X = (X-mean) / tf.sqrt(std+eps)

        if g is not None and b is not None:
            g = tf.reshape(g, [1,1,1,-1])
            b = tf.reshape(b, [1,1,1,-1])
            X = X*g + b

    elif X.get_shape().ndims == 2:
        mean = tf.reduce_mean(X, 0)
        std = tf.reduce_mean(tf.square(X-mean), 0)
        X = (X-mean) / tf.sqrt(std+eps)

        if g is not None and b is not None:
            g = tf.reshape(g, [1,-1])
            b = tf.reshape(b, [1,-1])
            X = X*g + b

    else:
        raise NotImplementedError

    return X

def layer_norm(X, eps=1e-8, g=None, b=None):
    if X.get_shape().ndims == 4:
        mean, std = tf.nn.moments(X, [1,2,3], keep_dims=True)   #shape=(batch_size, 1,1,1)
        X = (X-mean) / tf.sqrt(std+eps)

        if g is not None and b is not None:
            g = tf.reshape(g, [1,1,1,-1])
            b = tf.reshape(b, [1,1,1,-1])
            X = X*g + b

    elif X.get_shape().ndims == 2:
        mean = tf.reduce_mean(X, 0)
        std = tf.reduce_mean(tf.square(X-mean), 0)
        X = (X-mean) / tf.sqrt(std+eps)

        if g is not None and b is not None:
            g = tf.reshape(g, [1,-1])
            b = tf.reshape(b, [1,-1])
            X = X*g + b

    else:
        raise NotImplementedError

    return X

# --- [ Unrolled GAN (https://github.com/poolio/unrolled_gan)
def extract_update_dict(update_ops):
    """
    Extract variables and their new values from Assign and AssignAdd ops.

    Args:
        update_ops: list of Assign and AssignAdd ops, typically computed using Keras' opt.get_updates()

    Returns:
        dict mapping from variable values to their updated value
    """
    name_to_var = {v.name: v for v in tf.global_variables()}
    updates = {}
    for update in update_ops:
        var_name = update.op.inputs[0].name
        var = name_to_var[var_name]
        value = update.op.inputs[1]
        if update.op.type == 'Assign':
            updates[var.value()] = value
        elif update.op.type == 'AssignAdd':
            updates[var.value()] = var + value
        else:
            raise ValueError("Update op type (%s) must be of type Assign or AssignAdd"%update_op.op.type)
    return updates

def remove_original_op_attributes(graph):
    """Remove _original_op attribute from all operations in a graph."""
    for op in graph.get_operations():
        op._original_op = None

def graph_replace(*args, **kwargs):
    """Monkey patch graph_replace so that it works with TF 1.0"""
    _graph_replace = tf.contrib.graph_editor.graph_replace
    remove_original_op_attributes(tf.get_default_graph())
    return _graph_replace(*args, **kwargs)

def linear_interpolation(x, y, num=10):
    assert x.shape == y.shape
    assert num >= 2
    shift = 1./(num-1)
    interpolater = [x*(1-i*shift) + y*(i*shift) for i in range(num)]
    return np.array(interpolater)

###TODO###
def spherical_interpolation(x, y, num=10):
    assert x.shape == y.shape
    assert num >= 2
    shift = 1. / (num-1)
    x_norm = x / np.linalg.norm(x)
    y_norm = y / np.linalg.norm(y)
    dot = np.dot(x_norm, y_norm.T)
    if np.fabs(dot) > 0.9995:
        return linear_interpolation(x, y, num)
    if dot < 0.0:
        y_norm = y_norm
        dot = -dot
    theta = np.arccos(dot)
    unit_interpolater = [x_norm*np.cos(theta*i*shift) + (y_norm-x_norm*dot)*np.sin(theta*i*shift) for i in range(num)]
    return unit_interpolater
