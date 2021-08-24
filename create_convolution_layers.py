def create_convolution_layers(input_img):
  
  from tensorflow import keras
  from tensorflow.keras import layers  
  model = layers.Conv2D(32, (3, 3), padding='same', input_shape=(100,100,1))(input_img)
  model = layers.LeakyReLU(alpha=0.1)(model)
  model = layers.MaxPooling2D((2, 2),padding='same')(model)
  model = layers.Dropout(0.25)(model)
  
  model = layers.Conv2D(64, (3, 3), padding='same')(model)
  model = layers.LeakyReLU(alpha=0.1)(model)
  model = layers.MaxPooling2D(pool_size=(2, 2),padding='same')(model)
  model = layers.Dropout(0.25)(model)
    
  model = layers.Conv2D(128, (3, 3), padding='same')(model)
  model = layers.LeakyReLU(alpha=0.1)(model)
  model = layers.MaxPooling2D(pool_size=(2, 2),padding='same')(model)
  model = layers.Dropout(0.4)(model)
    
  return model

