#!/usr/bin/python

################################## Getting started ############################################
import tensorflow as tf
import numpy as np

sess = tf . Session ()
################################## Task 2 ############################################


def task2_fx(mat, x, b):
    const_A = tf.constant(mat, dtype = tf.float32, shape=[2,2])
    const_x = tf.constant(x, dtype = tf.float32, shape=[2,1])
    const_b = tf.constant(b, dtype = tf.float32, shape=[2,1])
    return  tf.matmul(const_A ,const_x) + const_b


def task2_gx(y_list):
    const_ylist = tf.constant(y_list, dtype=tf.float32)
    return tf.sin(const_ylist)

op_fx = task2_fx([1,2,3,4], 2, 5)
print("Task2 fx: ", sess.run(op_fx))

op_gx = task2_gx([1,2,3,4])
print("Task2 gx: ", sess.run(op_gx))

summary_writer = tf.summary.FileWriter(logdir='./', graph=sess.graph)


