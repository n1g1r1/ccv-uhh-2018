import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ** CONV1
# CONV1: 32 filters, 3-by-3, stride 1, ReLU
conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        stride=1,
        activation=tf.nn.relu)

# ** CONV2
# CONV1: 32 filters, 3-by-3, stride 1, ReLU
conv2 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        stride=1,
        activation=tf.nn.relu)

# ** POOL1
# POOL1: Max pool, 2-by-2, stride 2
pool1 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2)

# ** CONV3
# CONV3: 64 filters, 3-by-3, stride 1, ReLU
conv3 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        stride=1,
        activation=tf.nn.relu)


# ** CONV4
# CONV4: 64 filters, 3-by-3, stride 1, ReLU
conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        stride=1,
        activation=tf.nn.relu)


# ** POOL2
# POOL2: Max pool, 2-by-2, stride 2
pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2)


# ** CONV5
# CONV6: 64 filters, 3-by-3, stride 1, ReLU
conv5 = tf.layers.conv2d(
        inputs=pool2,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        stride=1,
        activation=tf.nn.relu)



# ** CONV6
# CONV7: 64 filters, 3-by-3, stride 1, ReLU
conv6 = tf.layers.conv2d(
        inputs=conv5,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        stride=1,
        activation=tf.nn.relu)


# ** POOL3
# POOL3: Max pool, 2-by-2, stride 2
pool3 = tf.layers.max_pooling2d(
        inputs=conv6,
        pool_size=[2, 2],
        strides=1,
        padding="same")



# ** CONV6
# CONV8: 128 filters, 3-by-3, stride 1, ReLU
conv7 = tf.layers.conv2d(
        inputs=conv5,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        stride=1,
        activation=tf.nn.relu)


# ** CONV7
# CONV8: 128 filters, 3-by-3, stride 1, ReLU
conv7 = tf.layers.conv2d(
        inputs=conv5,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        stride=1,
        activation=tf.nn.relu)


# ** CONV8
# CONV8: 128 filters, 3-by-3, stride 1, ReLU
conv8 = tf.layers.conv2d(
        inputs=conv5,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        stride=1,
        activation=tf.nn.relu)


multilevel_features = tf.concat([pool1, pool2, pool3, conv8], 3)

# ** CONV6
mf_conv1 = tf.layers.conv2d(
        inputs=multilevel_features,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        stride=1,
        activation=tf.nn.relu)

# ** CONV6
mf_conv2 = tf.layers.conv2d(
        inputs=mf_conv1,
        filters=1,
        kernel_size=[1, 1],
        padding="same",
        stride=1,
        activation=tf.nn.relu)


tf.image.resize_images(
        mf_conv2,
        [224,224],
        method=ResizeMethod.BILINEAR,
        align_corners=False)





