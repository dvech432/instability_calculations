
def cnn_classification_multiple_grid_search(lr, dc,op_type):

  from tensorflow import keras
  from tensorflow.keras import layers
  from create_convolution_layers import create_convolution_layers
  from keras.models import Model
  lin_input = layers.Input(shape=(100,100,1))
  lin_model = create_convolution_layers(lin_input)

  log_input = layers.Input(shape=(100,100,1))
  log_model = create_convolution_layers(log_input)

  conv = layers.concatenate([lin_model, log_model])

  conv = layers.Flatten()(conv)

  dense = layers.Dense(512)(conv)
  dense = layers.LeakyReLU(alpha=0.1)(dense)
  dense = layers.Dropout(0.5)(dense)

  output = layers.Dense(1, activation='sigmoid')(dense)

  m = keras.models.Model(inputs=[lin_input, log_input], outputs=[output])

  #### testing with various learning and decay rates
  if op_type=='Adam':
    opt = keras.optimizers.Adam(learning_rate=lr, decay=dc)
  if op_type=='SGD':
    opt = keras.optimizers.SGD(learning_rate=lr, decay=dc)  
  if op_type=='RMSprop':
    opt = keras.optimizers.RMSprop(learning_rate=lr, decay=dc)    
    
  m.compile(optimizer = opt, loss = 'binary_crossentropy',metrics='accuracy')
  
  #m.compile(optimizer = 'SGD', loss = 'binary_crossentropy',metrics='accuracy')
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
  
# model = keras.Sequential()
# #model.add(Dense(256, activation='relu', input_dim=366))
# model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape = (100,100,1)))
# #model.add(Conv2D(128, (3, 3), activation='relu'))
# #model.add(Conv2D(64, (3, 3), init='uniform'))
# model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# model.add(Flatten())

# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dropout(0.1))

# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dense(128, activation='relu'))

# model.add(layers.Dense(1, activation='linear'))

# model.compile(optimizer = 'adam', loss = 'mean_squared_error')