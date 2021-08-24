
def cnn_classification_multiple():

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

  model = keras.models.Model(inputs=[lin_input, log_input], outputs=[output])

  opt = keras.optimizers.Adam()

  model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
  
  return model

######################