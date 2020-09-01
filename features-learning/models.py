import tensorflow as tf
import numpy as np

# Some configuration parameters for convolution and dense layers
# This will avoid unnecessary repeats
# Be careful when you change those values as many models will overwrite
# some fields
conv_args = {
    "padding": "VALID",
    "biases_initializer": None,
    "activation_fn": None,
    "weights_initializer": tf.random_normal_initializer(0, 0.05)
}
dense_args = {
    "biases_initializer": None,
    "activation_fn": tf.nn.tanh,
    "weights_initializer": tf.random_normal_initializer(0, 0.05)
}

bn_args = {
    "decay": 0.,
    "center": True,
    "scale": False,
    "epsilon": 1e-8,
    "activation_fn": tf.nn.elu,
    "is_training": False
}


def mlp(c_i, out_num):
    with tf.variable_scope('evo', reuse=tf.AUTO_REUSE):
        c_i = tf.layers.dense(c_i, units=64, use_bias = True,activation = tf.nn.tanh,kernel_initializer=tf.random_normal_initializer(0, 0.05))
        c_i = tf.layers.dense(c_i, units=out_num, use_bias = True,activation =None,kernel_initializer=tf.random_normal_initializer(0, 0.05))
        c_i = c_i + 0.02*tf.random_normal([out_num])

    return c_i
