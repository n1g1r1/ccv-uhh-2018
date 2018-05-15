import matplotlib.pyplot as plt
### >>> COPIED CODE
import tensorflow as tf 
import numpy as np



### SETTINGS
batchsize   = 100
lr          = 0.1
alpha       = 1.1


### DATA


######################################################
#### Base CNN
######################################################

# The first dimension (the batch size) will be determined at runtime. 
x = tf.placeholder(tf.float32, shape=(None, 784))
labels = tf.placeholder(tf.int32, shape=(None,))

# Before feeding the images to the CNN, reshape them to proper size. # Meaning of -1: infer the actual value from the shape of the input
input_layer = tf.reshape(x, [-1, 224, 224, 3])

##################### BLOCK 1


# ** CONV1
# Input Tensor Shape: [batch_size, 224, 224, 1]
# Output Tensor Shape: [batch_size, 224, 224, 32] 
conv1 = tf.layers.conv2d(
    input_layer,
    filters=32,
    kernel_size=[3,3],
    padding="same",
    activation=tf.nn.relu
)

# ** CONV2
# Input Tensor Shape: [batch_size, 224, 224, 32]
# Output Tensor Shape: [batch_size, 224, 224, 32] 
conv2 = tf.layers.conv2d(
    conv1,
    filters=32,
    kernel_size=[3,3],
    padding="same",
    activation=tf.nn.relu
)


# ** POOL1
# Input Tensor Shape: [batch_size, 224, 224, 32]
# Output Tensor Shape: [batch_size, 112, 112, 32] 
pool1 = tf.layers.max_pooling2d(
    conv2,
    padding="same",
    pool_size=[2,2],
    strides=2
)

##################### BLOCK 2


# ** CONV3
# Input Tensor Shape: [batch_size, 112, 112, 32]
# Output Tensor Shape: [batch_size, 112, 112, 64]
conv3 = tf.layers.conv2d(
    pool1,
    filters=64,
    padding="same",
    kernel_size=[3,3],
    activation=tf.nn.relu
)

# ** CONV4
# Input Tensor Shape: [batch_size, 112, 112, 32]
# Output Tensor Shape: [batch_size, 112, 112, 64]
conv4 = tf.layers.conv2d(
    conv3,
    filters=64,
    padding="same",
    kernel_size=[3,3],
    activation=tf.nn.relu
)


# ** POOL2
# Input Tensor Shape: [batch_size, 112, 112, 64] 
# Output Tensor Shape: [batch_size, 56, 56, 64]
pool2 = tf.layers.max_pooling2d(
    conv4,
    padding="same",
    pool_size=[2,2],
    strides=2,
)


##################### BLOCK 3

# ** CONV5
# Input Tensor Shape: [batch_size, 56, 56, 64]
# Output Tensor Shape: [batch_size, 56, 56, 128]
conv5 = tf.layers.conv2d(
    pool2,
    filters=128,
    padding="same",
    kernel_size=[3,3],
    activation=tf.nn.relu
)


# ** CONV6
# Input Tensor Shape: [batch_size, 56, 56, 128]
# Output Tensor Shape: [batch_size, 56, 56, 128]
conv6 = tf.layers.conv2d(
    conv5,
    filters=128,
    padding="same",
    kernel_size=[3,3],
    activation=tf.nn.relu
)

# ** POOL3
# Input Tensor Shape: [batch_size, 56, 56, 128] 
# Output Tensor Shape: [batch_size, 56, 56, 128]
pool3 = tf.layers.max_pooling2d(
    conv6,
    padding="same",
    pool_size=[2,2],
    strides=1,
)

##################### BLOCK 4

# ** CONV7
# Input Tensor Shape: [batch_size, 56, 56, 128]
# Output Tensor Shape: [batch_size, 56, 56, 128]
conv7 = tf.layers.conv2d(
    pool3,
    filters=128,
    padding="same",
    kernel_size=[3,3],
    activation=tf.nn.relu
)


# ** CONV8
# Input Tensor Shape: [batch_size, 56, 56, 128]
# Output Tensor Shape: [batch_size, 56, 56, 128]
conv8 = tf.layers.conv2d(
    conv7,
    filters=128,
    padding="same",
    kernel_size=[3,3],
    activation=tf.nn.relu
)




### Multi-level feature representation

# Input Tensor Shapes:
# POOL2 shape [batch_size, 56, 56, 64]
# POOL3 shape  [batch_size, 56, 56, 128]
# CONV8 shape [batch_size, 56, 56, 128]

# Output Tensor Shape:
# [batch_size, 56, 56, 320]
multilevel_features = tf.concat([pool2, pool3, conv8], 3)



### Fixation prediction

deconv1 = tf.layers.conv2d(
    multilevel_features,
    filters = 64,
    kernel_size=[3,3],
    activation=tf.nn.relu
)

deconv2 = tf.layers.conv2d(
    deconv1,
    filters = 1,
    kernel_size=[1,1],
    activation=tf.nn.relu
)

size = tf.constant([224, 224], dtype=tf.int32, name="size")

saliency_imgs = tf.image.resize_images(
    deconv2,
    size,
    method=ResizeMethod.BICUBIC,
    align_corners=False
)



### Loss function

def loss_function(G, P, alpha):

    loss_sum  = 0
    for g, p in zip(G,P):
        loss_sum += pow(1 / (1 - g) * ((p / max(p)) - g),2)
    
    return 1 / len(224 * 224) * loss_sum



y = tf.placeholder(shape=[None, 224, 224, 1])

loss = loss_function(saliency_imgs, y, alpha)