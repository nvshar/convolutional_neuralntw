


from PIL import Image

import numpy as np
import os

path ='/home/navneet/Downloads/cifar/train'     # location of cifar10 training data images (50000 images)
ims=os.listdir(path)
pix=[]
# extracting pixel values from images
for i in ims:
    im =Image.open(path+'/'+i,mode='r')
    pix_val = list(im.getdata())
    pix_v=np.reshape(pix_val,(3,32,32))
    pix.append(pix_v)

len(pix)
X_train=np.reshape(pix,(50000,3,32,32))         # depth comes first if we are using theano as backend

# extracting labels using the name of the training image
y=[]
for i in ims:
    y.append(i[-7:-4])

labels=np.unique(y)
c=0
for i in range(10):
    y=map(lambda x:c if x==labels[i] else x,y)  # using laels from 0 to 9 for different classes
    c=c+1
 
y_train=np.reshape(y,(50000,1))                 


import keras
from keras.utils import np_utils

X_train=X_train/255
y_train= np_utils.to_categorical(y_train,num_classes=10)

from keras.models import Sequential
from keras.layers import Conv2D ,MaxPool2D, Flatten,Dense

model=Sequential()
model.add(Conv2D(16,(5,5),activation='relu',input_shape=(50000,3,32,32)))
model.add(Conv2D(16,(5,5),activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(32,kernel_size=(5,5),activation='relu'))
model.add(Conv2D(32,kernel_size=(5,5),activation='relu'))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=500,epochs=10)