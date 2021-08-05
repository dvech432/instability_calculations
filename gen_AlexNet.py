
def gen_AlexNet():

  from tensorflow import keras
  from tensorflow.keras import layers
  
  AlexNet = keras.Sequential()

#1st Convolutional Layer
  AlexNet.add(layers.Conv2D(filters=96, input_shape=(100,100,1), kernel_size=(11,11), strides=(4,4), padding='same'))
  AlexNet.add(layers.BatchNormalization())
  AlexNet.add(layers.Activation('relu'))
  AlexNet.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

#2nd Convolutional Layer
  AlexNet.add(layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='same'))
  AlexNet.add(layers.BatchNormalization())
  AlexNet.add(layers.Activation('relu'))
  AlexNet.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

#3rd Convolutional Layer
  AlexNet.add(layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
  AlexNet.add(layers.BatchNormalization())
  AlexNet.add(layers.Activation('relu'))

#4th Convolutional Layer
  AlexNet.add(layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
  AlexNet.add(layers.BatchNormalization())
  AlexNet.add(layers.Activation('relu'))

#5th Convolutional Layer
  AlexNet.add(layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
  AlexNet.add(layers.BatchNormalization())
  AlexNet.add(layers.Activation('relu'))
  AlexNet.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

#Passing it to a Fully Connected layer
  AlexNet.add(layers.Flatten())
# 1st Fully Connected Layer
  AlexNet.add(layers.Dense(4096, input_shape=(32,32,3,)))
  AlexNet.add(layers.BatchNormalization())
  AlexNet.add(layers.Activation('relu'))
# Add Dropout to prevent overfitting
  AlexNet.add(layers.Dropout(0.4))

#2nd Fully Connected Layer
  AlexNet.add(layers.Dense(4096))
  AlexNet.add(layers.BatchNormalization())
  AlexNet.add(layers.Activation('relu'))
#Add Dropout
  AlexNet.add(layers.Dropout(0.4))

#3rd Fully Connected Layer
  AlexNet.add(layers.Dense(1000))
  AlexNet.add(layers.BatchNormalization())
  AlexNet.add(layers.Activation('relu'))
#Add Dropout
  AlexNet.add(layers.Dropout(0.4))

#Output Layer
  AlexNet.add(layers.Dense(1))
  AlexNet.add(layers.BatchNormalization())
  AlexNet.add(layers.Activation('sigmoid'))
  opt = keras.optimizers.SGD(learning_rate=0.001, decay=1e-6)
  #m.compile(optimizer = opt, loss = 'binary_crossentropy',metrics='accuracy')
  AlexNet.compile(optimizer = opt, loss = 'binary_crossentropy',metrics='accuracy')
  
  return AlexNet