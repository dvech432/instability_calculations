
def cnn():

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
  return m
