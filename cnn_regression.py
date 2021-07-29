
def cnn_regression():

  from tensorflow import keras
  from tensorflow.keras import layers
  
  m = keras.Sequential()
  m.add(layers.Conv2D(filters=8, kernel_size=3, input_shape=(100,100,1),padding="same" )  )
  
  m.add(layers.Activation('relu'))
  m.add(layers.BatchNormalization())
  m.add(layers.AvgPool2D(pool_size=(2,2),strides=2)    )
  m.add(layers.Conv2D(filters=16, kernel_size=3,padding="same" )  )

  m.add(layers.Activation('relu'))
  m.add(layers.BatchNormalization())
  m.add(layers.AvgPool2D(pool_size=(2,2),strides=2)    )
  m.add(layers.Conv2D(filters=32, kernel_size=3, padding="same" )  )

  m.add(layers.Activation('relu'))
  m.add(layers.BatchNormalization())
  m.add(layers.AvgPool2D(pool_size=(2,2),strides=2)    )
  m.add(layers.Conv2D(filters=64, kernel_size=3, padding="same" )  )

  m.add(layers.Activation('relu'))
  m.add(layers.BatchNormalization())
  m.add(layers.Dropout(0.2))
  m.add(layers.Flatten())
  m.add(layers.Dense(1, activation='linear'))
  opt = keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)
  m.compile(optimizer = opt, loss = 'mean_squared_error')
  return m


### redoing everything


    # model = keras.Sequential()
    # model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(100, 100, 1)))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation="relu"))

    # model.add(layers.Flatten())
    # model.add(layers.Dense(64, activation="relu"))
    # model.add(layers.Dense(1))
    # model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
    # model.fit(x_train, np.reshape(params[:,0],(-1,1)), batch_size=24,epochs=200, verbose=1)
    # core_par_pred=model.predict(x_train)
    
  #### redoing everything again
  
  ## regression again
  
model = keras.Sequential()
#model.add(Dense(256, activation='relu', input_dim=366))
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape = (100,100,1)))
#model.add(Conv2D(128, (3, 3), activation='relu'))
#model.add(Conv2D(64, (3, 3), init='uniform'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.1))

model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(1, activation='linear'))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')