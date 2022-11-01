#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.layers import Input,Dense,Conv2D,Conv2DTranspose,Flatten,Reshape
from keras.datasets import cifar10
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from keras.models import Model
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.keras import backend as k 
from PIL import Image
import os 


# In[2]:


def rgbtogray(rgb):
    """Convert from color image (RGB) to grayscale.
       Source: opencv.org
       grayscale = 0.299*red + 0.587*green + 0.114*blue
       Argument:
       rgb (tensor): rgb image
       Return:
      (tensor): grayscale image
     """
    return np.dot(rgb[...,:3],[0.299, 0.587, 0.114])


# In[3]:


# load the CIFAR10 data
(x_train,_),(x_test,_)=cifar10.load_data()


# In[4]:


# input image dimensions
# we assume data format "channels_last"
img_rows=x_train.shape[1]
img_cols=x_train.shape[2]
channels=x_train.shape[3]
# create saved_images folder
img_dir='saved_images'
save_dir=os.path.join(os.getcwd(),img_dir)
if not os.path.isdir(save_dir):
     os.makedirs(save_dir)


# In[5]:


# display the 1st 100 input images (color and gray)
imgs=x_test[:100]
imgs=imgs.reshape((10,10,img_rows,img_cols,channels))
imgs=np.vstack(np.hstack(i) for i in imgs)
plt.figure()
plt.axis('off')
plt.title('Test color images (Ground Truth)')
plt.imshow(imgs, interpolation='none')
plt.show()


# In[6]:


# convert color train and test images to gray
x_train_gray=rgbtogray(x_train)
x_test_gray=rgbtogray(x_test)


# In[7]:


# display grayscale version of test images
imgs = x_test_gray[:100]
imgs = imgs.reshape((10, 10, img_rows, img_cols))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('Test gray images (Input)')
plt.imshow(imgs, interpolation='none', cmap='gray')
plt.show()


# In[8]:


# normalize output train and test color images
x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255
# normalize input train and test grayscale images
x_train_gray=x_train_gray.astype('float32')/255
x_test_gray=x_test_gray.astype('float32')/255


# In[9]:


# reshape images to row x col x channel for CNN output/validation
x_train=x_train.reshape(x_train.shape[0],img_rows,img_cols,channels)
x_test=x_test.reshape(x_test.shape[0],img_rows,img_cols,channels)


# In[10]:


# reshape images to row x col x channel for CNN input
x_train_gray = x_train_gray.reshape(x_train_gray.shape[0], img_rows, img_cols, 1)
x_test_gray = x_test_gray.reshape(x_test_gray.shape[0], img_rows, img_cols, 1)


# In[11]:


# network parameters
input_shape=(img_rows,img_cols,1)
batch=32
kernel=3
latent_dim=256
layers_filter=[64,128,256]


# In[12]:


# build the autoencoder model
# first build the encoder model
inputs=Input(shape=input_shape,name='encoder_input')
X=inputs
# stack of Conv2D(32)-Conv2D(64)
for filters in layers_filter:
    X=Conv2D(filters=filters,
             kernel_size=kernel,
             activation='relu',
             strides=2,padding='same')(X)


# In[13]:


# shape info needed to build decoder model
# so we don't do hand computation
# the input to the decoder's first
# Conv2DTranspose will have this shape
# shape is (4,4,256) which is processed by
# the decoder back to (32,32,3)
shape=k.int_shape(X)


# In[14]:


# generate latent vector
X=Flatten()(X)
latent=Dense(latent_dim,name='latent_vector')(X)


# In[15]:


#instantiate encoder model
encoder=Model(inputs,latent)


# In[16]:


# build the decoder model
latent_inputs=Input(shape=(latent_dim,),name='decoder_input')


# In[17]:


# use the shape (4,4,256) that was earlier saved
X=Dense(shape[1]*shape[2]*shape[3])(latent_inputs)
# from vector to suitable shape for transposed conv
X=Reshape((shape[1],shape[2],shape[3]))(X)


# In[18]:


# stack of Conv2DTranspose(256)-Conv2DTranspose(128)-Conv2DTranspose(64)
for filters in layers_filter[::-1]:
    X=Conv2DTranspose(filters=filters,
                      kernel_size=kernel,
                      activation='relu',
                      strides=2,padding='same')(X)


# In[19]:


# reconstruct the input
outputs=Conv2DTranspose(filters=3,kernel_size=kernel,
                        padding='same',activation='sigmoid',
                        name='decoder_outputs')(X)


# In[20]:


# instantiate decoder model
decoder=Model(latent_inputs,outputs,name='decoder')


# In[21]:


autoencoder=Model(inputs,decoder(encoder(inputs)),name='autoencoder')


# In[22]:


autoencoder.summary()


# In[23]:


# prepare model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'colorized_ae_model.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
     os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)


# In[24]:


# reduce learning rate by sqrt(0.1) if the loss does not improve in 5 epochs
lr_reduce=ReduceLROnPlateau(factor=np.sqrt(.1),
                            cooldown=0,patience=5,
                            verbose=1,min_lr=0.5e-6)


# In[25]:


# save weights for future use (e.g. reload parameters w/o training)
checkpoint=ModelCheckpoint(filepath=filepath,
                          monitor='val_loss',
                          verbose=1,
                          save_best_only=True)


# In[26]:


# Mean Square Error (MSE) loss function, Adam optimizer
autoencoder.compile(loss='mse', optimizer='adam')


# In[27]:


# called every epoch
callbacks=[lr_reduce,checkpoint]


# In[28]:


# train the autoencoder
autoencoder.fit(x_train_gray,x_train,
               validation_data=[x_test_gray,x_test],
               batch_size=batch,
               epochs=20,
               callbacks=callbacks)


# In[ ]:


# predict the autoencoder output from test data
x_decoded=autoencoder.predict(x_test_gray)


# In[ ]:


# display the 1st 100 colorized images
imgs = x_decoded[:100]
imgs = imgs.reshape((10, 10, img_rows, img_cols, channels))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('Colorized test images (Predicted)')
plt.imshow(imgs, interpolation='none')
plt.show()

