# import packages

from conf import myConfig as config
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv2D,Activation,Input,Add,MaxPooling2D,UpSampling2D,Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow.keras.backend as K
import numpy as np
from numpy import array
from numpy.linalg import norm
import tensorflow as tf
from numpy import *
import random
import cv2
from skimage.util import random_noise
import os
os.environ["CUDA_VISIBLE_DEVICES"]="5"
tf_device='/gpu:5'

# create CNN model
input_img=Input(shape=(None,None,1))

x=Conv2D(64,(3,3), dilation_rate=1,padding="same")(input_img)
x1=Activation('relu')(x)

x=UpSampling2D(interpolation="bilinear")(x1)

x=Conv2D(64,(3,3), dilation_rate=2,padding="same")(x)
x=Activation('relu')(x)

x=Conv2D(64,(3,3), dilation_rate=3,padding="same")(x)
x=Activation('relu')(x)

x=Conv2D(64,(3,3), dilation_rate=2,padding="same")(x)
x=Activation('relu')(x)
x=MaxPooling2D()(x)

x2 = Add()([x, x1])

x=MaxPooling2D()(x1)

x=Conv2D(64,(3,3), dilation_rate=2,padding="same")(x)
x=Activation('relu')(x)

x=Conv2D(64,(3,3), dilation_rate=3,padding="same")(x)
x=Activation('relu')(x)

x=Conv2D(64,(3,3), dilation_rate=2,padding="same")(x)
x=Activation('relu')(x)
x=UpSampling2D()(x)

x3 = Add()([x, x1])

x4 = Add()([x2, x3])

x = Conv2D(64,(3,3), padding="same")(x4)
x=Activation('relu')(x)

x5 = Add()([x, x1])

ebi=Conv2D(1,(3,3),padding="same")(x5)

x=Conv2D(64,(3,3), dilation_rate=1,padding="same")(ebi)
x=Activation('relu')(x)

x=Conv2D(64,(3,3), dilation_rate=3,padding="same")(x)
x=Activation('relu')(x)
x=MaxPooling2D(strides=2)(x)

x=Conv2D(64,(3,3), dilation_rate=5,padding="same")(x)
x_eb=Activation('relu')(x)

x=Conv2D(64,(3,3), dilation_rate=1,padding="same")(input_img)
x=Activation('relu')(x)

x=Conv2D(64,(3,3), dilation_rate=3,padding="same")(x)
x=Activation('relu')(x)
x=MaxPooling2D(strides=2)(x)

x=Conv2D(64,(3,3), dilation_rate=5,padding="same")(x)
x=Activation('relu')(x)

x11 = Concatenate()([x,x_eb])

x = UpSampling2D()(x11)

x=Conv2D(128,(3,3), dilation_rate=3,padding="same")(x)
x=Activation('relu')(x)

x=Conv2D(128,(3,3), dilation_rate=1,padding="same")(x)
x=Activation('relu')(x)

x7=Conv2D(1,(3,3),padding="same")(x)
model = Model(inputs=input_img, outputs=x7)

# load the data and normalize it
cleanImages=np.load(config.data)
print(cleanImages.dtype)
cleanImages=cleanImages/255.0
cleanImages=cleanImages.astype('float32')

# define augmentor and create custom flow
aug = ImageDataGenerator(rotation_range=30, fill_mode="nearest")

def myFlow(generator,X):
    for batch in generator.flow(x=X,batch_size=config.batch_size,seed=0):
        m,n,o,p=batch.shape
        looks=random.randint(1,21)
        stack=np.zeros((m,n,o,p,looks)) 
        for j in range(0,looks):
            stack[:,:,:,:,j] = random_noise(batch, mode='speckle') 
        noisyImagesBatch=np.mean(stack,axis=4)
        yield(noisyImagesBatch,batch)

# create custom learning rate scheduler
def lr_decay(epoch):
    initAlpha=0.0001
    factor=0.5
    dropEvery=30
    alpha=initAlpha*(factor ** np.floor((1+epoch)/dropEvery))
    return float(alpha)
callbacks=[LearningRateScheduler(lr_decay)]

# create custom loss, compile the model
print("[INFO] compilingTheModel")
opt=optimizers.Adam(learning_rate=0.0001)
def custom_loss(y_true,y_pred):
    diff=K.abs(y_true-y_pred)
    l1=(diff)/(config.batch_size)
    return l1
model.compile(loss=custom_loss,optimizer=opt)

# train
model.fit_generator(myFlow(aug,cleanImages),
epochs=config.epochs,steps_per_epoch=len(cleanImages)//config.batch_size,callbacks=callbacks,verbose=1)

# save the model
model.save('SIFSDNet_Synthetic.h5')
