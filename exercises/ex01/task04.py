import tensorflow as tf
import numpy as np
import cv2
import matplotlib . pyplot as plt
from PIL import Image

def get_gaussian_kernel(sigma,size=None):
    if not size:
        size=int(round(sigma*3*2+1))|1
    C = cv2.getGaussianKernel(size,sigma)
    C = np.outer(C,C).astype(np.float32)
    C /= np.sum(C)
    return C

img=Image.open('cameraman.png')
img_arr=np.array(img).astype(np.float32)/255.0
imgplot=plt.imshow(img_arr,cmap='gray')
plt.show()
