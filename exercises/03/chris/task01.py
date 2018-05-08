
import matplotlib.pyplot as plt
### >>> COPIED CODE
import tensorflow as tf 
import numpy as np

### SETTINGS
batchsize   = 100
lr          = 0.1


### DATA
mnist = tf.contrib.learn.datasets.load_dataset("mnist") 
train_data = mnist.train.images 
# Returns np.array 
train_labels = np.asarray(mnist.train.labels , dtype=np.int32) 
eval_data = mnist.test.images # Returns np.array
eval_labels = np.asarray(mnist.test.labels , dtype=np.int32)

### COPIED CODE <<<

def mnist_dataset():

    print('''
     -----------------------
    | Task 1: MNIST Dataset |
     -----------------------
    ''')

    print('Training data shape: ' + str(train_data.shape)) 
    print('Training labels shape: ' + str(train_labels.shape)) 
    print('Evaluation data shape: ' + str(eval_data.shape)) 
    print('Evaluation labels shape: ' + str(eval_labels.shape))

    ### COPIED CODE <<<

    # How many training examples are there? How many evaluation examples are there?
    print('Training examples: {}\nEvaluation examples: {}'.format(len(train_data), len(eval_data)))

    # Why is the size of the second dimension of train_data and eval_data 784?
    print('''
    Q: Why is the size of the second dimension of train_data and eval_data 784?
    A: Because the image has the size of 28 * 28 = 784 pixels, stored in a row.
    ''')

    # Draw first trainng image
    img = train_data[0,:].reshape(28,28)
    imgplot = plt.imshow(img, cmap='gray')
    plt.show()

    # Which digit is in the image, what's the corresponding label
    print("The first digit is {}".format(train_labels[0]))





### SETUP 

print('''
    ------------------------
| Task 2: MNIST Dataset  |
    ------------------------
''')

# Placeholders for input images and labels

# The first dimension (the batch size) will be determined at runtime. 
x = tf.placeholder(tf.float32, shape=(None, 784))
labels = tf.placeholder(tf.int32, shape=(None,))

# Before feeding the images to the CNN, reshape them to proper size. # Meaning of -1: infer the actual value from the shape of the input
input_layer = tf.reshape(x, [-1, 28, 28, 1])

######### Next: insert CNN layers below!

# Why are the layer i/o tensor shapes as they are claimed to be?
# Compare the shapes to yours to see if your implementation is correct

# ** CONV1
# Input Tensor Shape: [batch_size, 28, 28, 1]
# Output Tensor Shape: [batch_size, 28, 28, 32] 
conv1 = tf.layers.conv2d(
    input_layer,
    filters=32,
    kernel_size=[5,5],
    padding="same",
    activation=tf.nn.relu
)


# ** POOL1
# Input Tensor Shape: [batch_size, 28, 28, 32]
# Output Tensor Shape: [batch_size, 14, 14, 32] 
pool1 = tf.layers.max_pooling2d(
    conv1,
    pool_size=[2,2],
    strides=2
)


# ** CONV2
# Input Tensor Shape: [batch_size, 14, 14, 32]
# Output Tensor Shape: [batch_size, 14, 14, 64]
conv2 = tf.layers.conv2d(
    pool1,
    filters=64,
    padding="same",
    kernel_size=[5,5],
    activation=tf.nn.relu
)


# ** POOL2

# Input Tensor Shape: [batch_size, 14, 14, 64] 
# Output Tensor Shape: [batch_size, 7, 7, 64]
pool2 = tf.layers.max_pooling2d(
    conv2,
    pool_size=[2,2],
    strides=2,
)


# ** Flatten tensor into a batch of vectors
# Input Tensor Shape: [batch_size, 7, 7, 64]
# Output Tensor Shape: [batch_size, 7 * 7 * 64] 
pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])


# ** FC1
# Input Tensor Shape: [batch_size, 7 * 7 * 64]
# Output 
fc1 = tf.layers.dense(pool2_flat, 
    units=1024, 
    activation=tf.nn.relu
)


# ** FC2
# Input
# Output
logits = tf.layers.dense(fc1, units=10)


# Define a loss function to optimize.
# This assumes the final layer of your CNN is called logits.
loss = tf.losses.sparse_softmax_cross_entropy(labels=labels , logits=logits)
# Accuracy is the fraction of correctly predicted classes.
predictions = tf.argmax(logits , axis=1, output_type=tf.int32) 
correct_prediction = tf.equal(labels , predictions)
accuracy = tf.reduce_mean(tf.cast(correct_prediction , tf.float32))
# Define an optimizer and training op
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr) 
train_op = optimizer.minimize(loss)    



### Train CNN
num_training_examples = train_data.shape[0] 
num_batches = int(num_training_examples / batchsize)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for b in range(num_batches):
        starting_index = b*batchsize
        stopping_index = min((b+1)*batchsize , num_training_examples)

        batch_images = train_data[starting_index:stopping_index , :] 
        batch_labels = train_labels[starting_index:stopping_index]

        batch_loss, batch_acc, _ = sess.run([loss, accuracy, train_op], feed_dict={x: batch_images , labels: batch_labels})
        print('Batch ' + str(b) + ', loss: ' + str(batch_loss) + ', accuracy: ' + str(batch_acc))




    #### Validation

    num_eval_examples = eval_data.shape[0] 
    num_eval_batches = int(num_eval_examples / batchsize)
    total_eval_loss = 0.0
    total_eval_acc = 0.0
    for b in range(num_eval_batches):
        starting_index = b*batchsize
        stopping_index = min((b+1)*batchsize , num_eval_examples)
        batch_images = eval_data[starting_index:stopping_index , :] 
        batch_labels = eval_labels[starting_index:stopping_index]
        [batch_eval_loss, batch_eval_acc] = sess.run([loss, accuracy], feed_dict={x: batch_images , labels: batch_labels})
        total_eval_loss += batch_eval_loss 
        total_eval_acc += batch_eval_acc


    print('''
    
    ----------------------------------------------
    Training results: 

    Average accuracy on validation data: {}
    Average loss on validation data: {}
    '''.format(total_eval_acc / num_eval_batches, total_eval_loss / num_eval_batches))

    summary_writer = tf.summary.FileWriter(logdir='cnn-summary/', graph=sess.graph)