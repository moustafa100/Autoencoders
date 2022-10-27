#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.layers import Input,Dense,Conv2D,Conv2DTranspose,Flatten,Reshape
from keras.datasets import mnist
from keras.models import Model
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.keras import backend as k 
from PIL import Image


# In[2]:


# load MNIST dataset
(x_train,_),(x_test,_)=mnist.load_data()


# In[3]:


# reshape to (28, 28, 1) and normalize input images
img_size=x_train.shape[1]
x_train=np.reshape(x_train,[-1,img_size,img_size,1])
x_test=np.reshape(x_train,[-1,img_size,img_size,1])
x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255


# In[4]:


# generate corrupted MNIST images by adding noise with normal dist
# centered at 0.5 and std=0.5
noisy=np.random.normal(loc=.5,scale=.5,size=x_train.shape)
x_train_noisy=x_train+noisy
noise=np.random.normal(loc=.5,scale=.5,size=x_test.shape)
x_test_noisy=x_test+noise


# In[5]:


# adding noise may exceed normalized pixel values>1.0 or <0.0
# clip pixel values >1.0 to 1.0 and <0.0 to 0.0
x_train_noisy=np.clip(x_train_noisy,0.,1.)
x_test_noisy=np.clip(x_test_noisy,0.,1.)


# In[6]:


# network parameters
input_size=(img_size,img_size,1)
batch=32
kernel=3
latent_dim=16


# In[7]:


# encoder/decoder number of CNN layers and filters per layer
layers_filter=[32,64]


# In[8]:


# build the autoencoder model
# first build the encoder model
inputs=Input(shape=input_size,name='encoder_input')
X=inputs
# stack of Conv2D(32)-Conv2D(64)
for filters in layers_filter:
    X=Conv2D(filters=filters,
             kernel_size=kernel,
             activation='relu',
             strides=2,padding='same')(X)


# In[9]:


# shape info needed to build decoder model
# so we don't do hand computation
# the input to the decoder's first
# Conv2DTranspose will have this shape
# shape is (7, 7, 64) which is processed by
# the decoder back to (28, 28, 1)
shape=k.int_shape(X)


# In[10]:


# generate latent vector
X=Flatten()(X)
latent=Dense(latent_dim,name='latent_vector')(X)


# In[11]:


#instantiate encoder model
encoder=Model(inputs,latent)


# In[12]:


# build the decoder model


# In[13]:


latent_inputs=Input(shape=(latent_dim,),name='decoder_input')


# In[14]:


# use the shape (7, 7, 64) that was earlier saved
X=Dense(shape[1]*shape[2]*shape[3])(latent_inputs)
# from vector to suitable shape for transposed conv
X=Reshape((shape[1],shape[2],shape[3]))(X)


# In[15]:


# stack of Conv2DTranspose(64)-Conv2DTranspose(32)
for filters in layers_filter[::-1]:
    X=Conv2DTranspose(filters=filters,
                      kernel_size=kernel,
                      activation='relu',
                      strides=2,padding='same')(X)


# In[16]:


# reconstruct the input
outputs=Conv2DTranspose(filters=1,kernel_size=kernel,
                        padding='same',activation='sigmoid',
                        name='decoder_outputs')(X)


# In[17]:


# instantiate decoder model
decoder=Model(latent_inputs,outputs,name='decoder')


# In[18]:


autoencoder=Model(inputs,decoder(encoder(inputs)),name='autoencoder')


# In[19]:


autoencoder.summary()


# In[21]:


#Mean Square Error (MSE) loss function, Adam optimizer
autoencoder.compile(loss='mse',optimizer='adam')


# In[23]:


# train the autoencoder
autoencoder.fit(x_train_noisy,x_train,
                validation_data=(x_test_noisy,x_test),
                epochs=10,batch_size=batch)


# In[24]:


# predict the autoencoder output from test data
x_decoded=autoencoder.predict(x_test_noisy)


# In[27]:


# 3 sets of images with 9 MNIST digits
# 1st rows - original images
# 2nd rows - images corrupted by noise
# 3rd rows - denoised images
rows, cols = 3, 9
num = rows * cols
imgs = np.concatenate([x_test[:num], x_test_noisy[:num], x_decoded[:num]])
imgs = imgs.reshape((rows * 3, cols, img_size, img_size))
imgs = np.vstack(np.split(imgs, rows, axis=1))
imgs = imgs.reshape((rows * 3, -1, img_size, img_size))
imgs = np.vstack([np.hstack(i) for i in imgs])
imgs = (imgs * 255).astype(np.uint8)
plt.figure()
plt.axis('off')
plt.title('Original images: top rows, '
 'Corrupted Input: middle rows, '
 'Denoised Input: third rows')
plt.imshow(imgs, interpolation='none', cmap='gray')
Image.fromarray(imgs).save('corrupted_and_denoised.png')
plt.show()


# In[ ]:




