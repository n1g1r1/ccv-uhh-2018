import tensorflow as tf

# SETTINGS ##############################################




# FUNCTIONS #############################################

def user_input():
    return input("Enter a number: \n")


# GRAPH #################################################

x = tf.placeholder(tf.float32, name="user_input")
n = tf.Variable(0.0, name="number_of_digits")
average = tf.Variable(0.0, name="average")

# Operations
init = tf.global_variables_initializer()
update_average = average.assign(x / n + ((n - 1) * average) / n)
increment_n = n.assign_add(1.0)


# SESSION ################################################

with tf.Session() as sess:

    # Initialize all variables
    sess.run(init)

    # Get digits
    while True:
        
        # Print the result
        print('Result -  n: {}, average_n-1: {}, y: {}'.format(sess.run(increment_n), sess.run(y), sess.run(update_average, feed_dict={x: user_input()})))