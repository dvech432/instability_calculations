

from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , ReLU, Activation, AvgPool2D, LeakyReLU, MaxPooling2D


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(100,100,1),padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))                  
#model.add(Dense(num_classes, activation='softmax'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')



model.fit(x_train, np.reshape(params[:,0],(-1,1)), batch_size=24,epochs=200, verbose=1)



core_ani_pred=model.predict(x_train)