#!/usr/bin/env python
# coding: utf-8

# In[7]:


from keras.layers import Input,Dense,Conv2D,Conv2DTranspose,Flatten,Reshape
from keras.datasets import mnist
from keras.models import Model
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.keras import backend as k 


# In[8]:


# load MNIST dataset
(x_train,_),(x_test,_)=mnist.load_data()


# In[11]:


# reshape to (28, 28, 1) and normalize input images
img_size=x_train.shape[1]
x_train=np.reshape(x_train,[-1,img_size,img_size,1])
x_test=np.reshape(x_train,[-1,img_size,img_size,1])
x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255


# In[12]:


# network parameters
input_size=(img_size,img_size,1)
batch=32
kernel=3
latent_dim=16


# In[13]:


# encoder/decoder number of CNN layers and filters per layer
layers_filter=[32,64]


# In[20]:


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
    


# In[21]:


# shape info needed to build decoder model
# so we don't do hand computation
# the input to the decoder's first
# Conv2DTranspose will have this shape
# shape is (7, 7, 64) which is processed by
# the decoder back to (28, 28, 1)
shape=k.int_shape(X)


# In[22]:


# generate latent vector
X=Flatten()(X)
latent=Dense(latent_dim,name='latent_vector')(X)


# In[23]:


# instantiate encoder model
encoder=Model(inputs,latent)
encoder.summary()


# In[24]:


# build the decoder model


# In[25]:


latent_inputs=Input(shape=(latent_dim,),name='decoder_input')


# In[31]:


# use the shape (7, 7, 64) that was earlier saved
X=Dense(shape[1]*shape[2]*shape[3])(latent_inputs)
# from vector to suitable shape for transposed conv
X=Reshape((shape[1],shape[2],shape[3]))(X)


# In[32]:


# stack of Conv2DTranspose(64)-Conv2DTranspose(32)
for filters in layers_filter[::-1]:
    X=Conv2DTranspose(filters=filters,
                      kernel_size=kernel,
                      activation='relu',
                      strides=2,padding='same')(X)


# In[34]:


# reconstruct the input
outputs=Conv2DTranspose(filters=1,kernel_size=kernel,
                        padding='same',activation='sigmoid',
                        name='decoder_outputs')(X)


# In[35]:


# instantiate decoder model
decoder=Model(latent_inputs,outputs,name='decoder')


# In[36]:


decoder.summary()


# In[37]:


autoencoder=Model(inputs,decoder(encoder(inputs)),name='autoencoder')


# In[38]:


autoencoder.summary()


# In[39]:


# Mean Square Error (MSE) loss function, Adam optimizer
autoencoder.compile(loss='mse',optimizer='adam')


# In[40]:


# train the autoencoder
autoencoder.fit(x_train,x_train,
                validation_data=(x_test,x_test),
                epochs=3,batch_size=batch)


# In[42]:


# predict the autoencoder output from test data
x_decoded=autoencoder.predict(x_test)


# In[44]:


# display the 1st 8 test input and decoded images
imgs = np.concatenate([x_test[:8], x_decoded[:8]])
imgs = imgs.reshape((4, 4, img_size, img_size))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('Input: 1st 2 rows, Decoded: last 2 rows')
plt.imshow(imgs, interpolation='none', cmap='gray')
plt.savefig('input_and_decoded.png')
plt.show()


# In[ ]:




