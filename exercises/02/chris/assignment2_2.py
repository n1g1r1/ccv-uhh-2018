import tensorflow as tf
import numpy as np

W = tf.Variable(tf.zeros([2,2], tf.float32))
x = tf.placeholder(tf.float32)
b = tf.Variable(tf.zeros([2,1], tf.float32))

init = tf.global_variables_initializer()
y = tf.layers.dense(x, k, activation=tf.nn.relu,  , use_bias=True, name="my_dense_layer")

with tf.Session() as sess:
    sess.run(init)

