import numpy as np
import tensorflow as tf

## Settings
lr = 0.5

## Weights
W = tf.Variable(tf.zeros((100,)), name="weights") # Vector of 100 zeros, b = 0
b = tf.Variable(tf.random_uniform((784, 100), -1, 1), name="biases") # Matrix with shape of 784x100 with values of -1, 1

## Input data (added in execution time)
x = tf.placeholder(tf.float32, (100, 784))

## Opterations

# Get the whole graph
tf.get_default_graph().get_operations()

## Loss node

# Output of NN
prediction = tf.nn.softmax(...) # TODO: to be defined

# Array of labels the network will get trained on
label = tf.placeholder(tf.float32, [None, 10])

# Cross entropy function to compute the errors
cross_entropy = tf.reduce_mean(-tf.reduce_sum(label *  tf.log(prediction), reduction_indices=[1]))

# Train the neural network by reducing the errors by backpropagation
train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)
# ReLu
# h = tf.nn.relu(tf.matmul(x, W) + b)

# Initialise session
session = tf.Session()

# Run the operations
session.run(tf.initialize_all_variables()) # Lazy evaluation, evaluates only at runtime
# session.run(h, {x: np.random.random(100,784)}) # Dict mapping of graph nodes to concrete values

for i in range(1000):
    batch_x, batch_label = data.next_batch()
    # Runs basically the training function whih have multiple operations in
    session.run(train_step, feed_dict = {x: batch_x, label: batch_label})
