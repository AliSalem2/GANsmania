from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.datasets.mnist import load_data
from IPython.display import clear_output
import os


from PIL import Image
from matplotlib import pyplot as plt 
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))



def inv_sigmoid(x):
    return np.log(y/(1-y))



%matplotlib inline
path = '/home/leesalem/code/AliSalem2/data-challenges/06-Deep-Learning/03-Convolutional-Neural-Networks/05-Autoencoder/data/Abstract_gallery'

os.getcwd()
img_list = os.listdir(path)


def access_images(img_list,path,length):
    #accessing images
    pixels = []
    imgs = []
    for i in range(length):
        img = Image.open(path+'/'+img_list[i],'r')
        basewidth = 100
        img = img.resize((basewidth,basewidth), Image.ANTIALIAS)
        pix = np.array(img.getdata())
        pixels.append(pix.reshape(100,100,3))
        imgs.append(img)
    return np.array(pixels),imgs



def show_image(pix_list):
    #showing image 
    array = np.array(pix_list.reshape(100,100,3), dtype=np.uint8)
    new_image = Image.fromarray(array)
    new_image.show()






def generate_real_samples(dataset, n_samples):
    #This calls upon the real and fake samples and generates the latent points that are used as the input for the generator.
    ix = randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = ones((n_samples, 1))
    return X, y
 
def generate_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


def generate_fake_samples(g_model, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    X = g_model.predict(x_input)
    y = zeros((n_samples, 1))
    return X, y