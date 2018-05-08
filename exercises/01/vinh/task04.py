import tensorflow as tf
import numpy as np
import cv2
import matplotlib . pyplot as plt
from PIL import Image

sess = tf.Session ()

def get_gaussian_kernel(sigma,size=None):
    if not size:
        size=int(round(sigma*3*2+1))|1
    C = cv2.getGaussianKernel(size,sigma)
    C = np.outer(C,C).astype(np.float32)
    C /= np.sum(C)
    return C

def show_cameraman():
    img=Image.open('cameraman.png')
    return np.array(img).astype(np.float32)/255.0

def apply_gaussian_filter(sigma):
    image=Image.open('cameraman.png')
    image_tf = tf.cast(image, tf.float32)

    gaussian_filter = tf.constant(get_gaussian_kernel(sigma))
    image_resized = tf.reshape(image_tf, [1,image_tf.shape[0], image_tf.shape[1],1])
    gaussian_filter_resized = tf.reshape(gaussian_filter, [gaussian_filter.shape[0],gaussian_filter.shape[1],1,1])
    op = tf.nn.conv2d(image_resized, gaussian_filter_resized, strides=[1, 1, 1, 1], padding='SAME')
    img = sess.run(op)
    return  np.reshape(img, [256,256])

def plot_before_after(img_before, img_after):
    fig = plt.figure()
    fig.add_subplot(1,2,1)
    plt.imshow(img_before)
    fig.add_subplot(1,2,2)
    plt.imshow(img_after)
    plt.show()

img_before = show_cameraman()
img_after = apply_gaussian_filter(sigma=10)

plot_before_after(img_before, img_after)

