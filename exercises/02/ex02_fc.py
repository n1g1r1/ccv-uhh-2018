import tensorflow as tf
k = 2 # output dimensionality

# Single 2-dimensional vector as input: x has shape [1, 2]
x = tf.constant([[1, -2]], tf.float32)

# FC layer definition with weights initialized to identity matrix and biases initialized to all ones
y = tf.layers.dense(inputs=x, 
					units=k, 
					activation=tf.nn.relu, 
					kernel_initializer=tf.initializers.identity, 
					bias_initializer=tf.ones_initializer())

with tf.Session() as sess:
	# Initializer all variables before running any other ops!
	sess.run(tf.global_variables_initializer())

	y_out = sess.run(y)
	print(y_out)