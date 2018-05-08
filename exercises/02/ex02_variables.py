import tensorflow as tf

# This number list will be fed in one number at a time.
number_list = [0.0, 2.0, 4.0, 6.0, 8.0]

# Define computational graph
x_input = tf.placeholder(tf.float32, shape=())
x_avg = tf.Variable(0.0, trainable=False, name='running_average')
n = tf.Variable(0.0, trainable=False, name='number_of_samples')

# When run, the next operation will increment value of n by 1.
increment_n = n.assign_add(1.0)
# When run, the next operation will update value of x_avg by Eq. (1) of the exercise sheet.
# Inside the assign call, all references to x_avg or n will take the current value of that variable.
update_avg = x_avg.assign( x_input/n + x_avg*(n-1) / n )

# Done defining the computational graph.
with tf.Session() as sess:
	# We initialize all variables before use
	sess.run([x_avg.initializer, n.initializer])
	for x in number_list:
		# We will input values 0, 2, 4, 6, 8; so always 2 times the current value of the loop variable.

		# First increment n.
		nv = sess.run(increment_n)
		# Then update the average and feed the new input value
		new_average = sess.run(update_avg, feed_dict={x_input: x})

		print('Last number input: ' + str(x) + ', running average is ' + str(new_average))