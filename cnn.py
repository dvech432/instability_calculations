# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 12:34:56 2021

@author: vechd
"""


from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , ReLU, Activation, AvgPool2D


# Initialize CNN
m = Sequential()
m.add(Conv2D(filters=8, kernel_size=3, input_shape=(100,100,3),padding="same" )  )

m.add( Activation('relu'))
m.add(AvgPool2D(pool_size=(2,2),strides=2)    )
m.add( Conv2D(filters=16, kernel_size=3, input_shape=(100,100,3),padding="same" )  )

m.add( Activation('relu'))
m.add(AvgPool2D(pool_size=(2,2),strides=2)    )
m.add( Conv2D(filters=32, kernel_size=3, input_shape=(100,100,3),padding="same" )  )

m.add( Activation('relu'))
m.add(AvgPool2D(pool_size=(2,2),strides=2)    )
m.add( Conv2D(filters=64, kernel_size=3, input_shape=(100,100,3),padding="same" )  )

m.add( Activation('relu'))
m.add(Dropout(0.2))
m.add(Dense(2, activation='softmax'))
m.compile(optimizer = 'sgd', loss = 'binary_crossentropy')


#layers = [
#    imageInputLayer([100 100 1])
#    convolution2dLayer(3,8,'Padding','same')

#    reluLayer
#    averagePooling2dLayer(2,'Stride',2)
#    convolution2dLayer(3,16,'Padding','same')

#    reluLayer
#    averagePooling2dLayer(2,'Stride',2)
#    convolution2dLayer(3,32,'Padding','same')

#    reluLayer
#    averagePooling2dLayer(2,'Stride',2)
#    convolution2dLayer(3,64,'Padding','same')

#    reluLayer
#    convolution2dLayer(3,64,'Padding','same')
 
#    reluLayer
#    dropoutLayer(0.2)
#    fullyConnectedLayer(3) %number of classes
#    softmaxLayer
#    classificationLayer];

#
# %%  57% after 1 epoch



# Fit LSTM model
# history = m.fit(xin, next_X, epochs = 50, batch_size = 50,verbose=0)
