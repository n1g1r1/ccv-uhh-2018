import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
### Setting up the data
a = 1.1
b = 0.2
n = 128 # number of points in dataset

# Values at linearly spaced x for plotting
xg = np.linspace(-1.0, 1.0, num=20)
yg = a*xg + b

# Dataset: added normally distributed noise to make it interesting
x = np.random.uniform(low=-1.0, high=1.0, size=n)
y = a*x + b + np.random.normal(loc=0.0, scale=0.4, size=n)

X = np.vstack([x, np.ones(len(x))]).T
a_hat, b_hat = np.linalg.lstsq(X, y)[0]

### Set up the computational graph
xp = tf.placeholder(tf.float32, shape=(n,1), name="x")
yp = tf.placeholder(tf.float32, shape=(n,1), name="y")

y_predicted = tf.layers.dense(xp, 1, activation=None, use_bias=True)
loss = tf.losses.mean_squared_error(labels=yp, predictions=y_predicted)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
minimize_op = optimizer.minimize(loss)

# Convenient way to get names of all trainable variables in graph
variables_names = [v.name for v in tf.trainable_variables()]

loss_summary = tf.summary.scalar(name="loss", tensor=loss)

### Running the computational graph
with tf.Session() as sess:
	summary_writer = tf.summary.FileWriter(logdir="./", graph=sess.graph)
	sess.run(tf.global_variables_initializer())
	for k in range(100): # Train for 100 steps...
		# Note: x and y have shape (n,). However, dense layer is expecting
		# an input of size (n,1). So we will expand a new dimension to both x and y
		# while feeding them as inputs.
		_, l = sess.run([minimize_op, loss_summary], {xp: np.expand_dims(x,1), 
							yp: np.expand_dims(y,1)})
		summary_writer.add_summary(l, global_step=k)

	trained_values = sess.run(variables_names)
	for varname, varvalue in zip(variables_names, trained_values):
	    # Store the learned values to variables with appropriate name
	    if "kernel" in varname: # if name contains "kernel", this is the parameter a
	    	a_sgd = float(varvalue)
	    else: # otherwise it's b
	    	b_sgd = float(varvalue)

### Examining the outputs
print('True values')
print(a, b)
print('OLS solution')
print(a_hat, b_hat)
print('SGD solution')
print(a_sgd, b_sgd)

# Visualize the data and solution via SGD and OLS
y_sgd = a_sgd*xg + b_sgd
y_ols = a_hat*xg + b_hat

plt.plot(x, y, 'bo')
line_true, = plt.plot(xg,yg,'r', label='True')
line_sgd, = plt.plot(xg,y_sgd,'g', label='SGD')
line_ols, = plt.plot(xg,y_ols,'c', label='OLS')
plt.ylabel('y')
plt.xlabel('x')
plt.legend(handles=[line_true, line_sgd, line_ols])
plt.show()