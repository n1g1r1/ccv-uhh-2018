import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images #Returnsnp.array
train_labels = np.asarray(mnist.train.labels,dtype = np.int32)
eval_data = mnist.test.images #Returnsnp.array
eval_labels = np.asarray(mnist.test.labels,dtype = np.int32)

print('Trainingdatashape:'+str(train_data.shape))
print('Traininglabelsshape:'+str(train_labels.shape))
print('Evaluationdatashape:'+str(eval_data.shape))
print('Evaluationlabelsshape:'+str(eval_labels.shape))


# Draw the first element of the training set. Its a 3.
def show_the_first_n_elements(n):
    for i in range(n):
        dim_a = int(math.sqrt(len(train_data[i])))
        image_np = np.reshape(train_data[i], [dim_a,dim_a])
        imgplot = plt.imshow(image_np,cmap = 'gray')
        plt.show()


# Placeholders for input images and labels
# The first dimension (the batch size ) will be determined at runtime .
x=tf.placeholder(tf.float32,shape=(None,784))
labels=tf.placeholder(tf.int32,shape=(None,))

# Before feeding the images to the CNN , reshape them to proper size .
# Meaning of -1: infer the actual value from the shape of the input
input_layer=tf.reshape(x,[-1,28,28,1])

# Next : insert CNN layers below !
# Why are the layer i/o tensor shapes as they are claimed to be?
# Compare the shapes to yours to see if your implementation is correct

# ** CONV1
# Input Tensor Shape : [ batch_size , 28 , 28 , 1]
# Output Tensor Shape : [ batch_size , 28 , 28 , 32]
conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

# ** POOL1
# Input Tensor Shape : [ batch_size , 28 , 28 , 32]
# Output Tensor Shape : [ batch_size , 14 , 14 , 32]
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

# ** CONV2
# Input Tensor Shape : [ batch_size , 14 , 14 , 32]
# Output Tensor Shape : [ batch_size , 14 , 14 , 64]
conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

# ** POOL2
# Input Tensor Shape : [ batch_size , 14 , 14 , 64]
# Output Tensor Shape : [ batch_size , 7 , 7 , 64]
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

# ** Flatten tensor into a batch of vectors
# Input Tensor Shape : [ batch_size , 7 , 7 , 64]
# Output Tensor Shape : [ batch_size , 7 * 7 * 64]
pool2_flat = tf.reshape(pool2,[-1,7*7*64])

# ** FC1
# Input Tensor Shape : [ batch_size , 7 * 7 * 64]
# Output Tensor Shape : [ batch_size , 1024]
fc1 = tf.layers.dense(pool2_flat, 1024, activation=tf.nn.relu)

# ** FC2
# Input Tensor Shape : [ batch_size , 1024]
# Output Tensor Shape : [ batch_size , 10]
logits = tf.layers.dense(fc1, 10)

# Define a loss function to optimize .
# This assumes the final layer of your CNN is called logits .
loss=tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)

 # Accuracy is the fraction of correctly predicted classes .
predictions=tf.argmax(logits,axis=1,output_type=tf.int32)
correct_prediction=tf.equal(labels,predictions)
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

 # Define an optimizer and training op
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001)
train_op=optimizer.minimize(loss)


###################### TRAIN THE CNN ##########################################

batchsize = 100
num_training_examples = train_data . shape [0]
num_batches=int(num_training_examples/batchsize)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for b in range ( num_batches ) :
        starting_index=b*batchsize
        stopping_index=min((b+1)*batchsize,num_training_examples)

        batch_images=train_data[starting_index:stopping_index,:]
        batch_labels=train_labels[starting_index:stopping_index]

        batch_loss,batch_acc,_=sess.run([loss,accuracy,train_op],feed_dict={x:batch_images,labels:batch_labels})
        print('Batch'+str(b)+',loss: ' +str(batch_loss)+ ', accuracy: '+str(batch_acc))

if __name__ == "__main__":
    # show_the_first_n_elements(10)
    print("hello")
