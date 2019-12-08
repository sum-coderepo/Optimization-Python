import os, time
import numpy as np
import tensorflow as tf # version 1.14
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import TensorBoard

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.expand_dims(X_train, 3)
X_test = np.expand_dims(X_test, 3)

optimizers = [
    'Adagrad',
    'Adam',
    'RMSprop',
    'SGD'
]
for optimizer in optimizers:
    model = Sequential()
    model.add(Conv2D(16, (3,3), activation='relu', input_shape=X_train[0].shape))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, (3,3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    tensorboard = tf.keras.callbacks.TensorBoard(os.path.join('log_test1',f'{optimizer}_{time.time()}'))
    model.fit(X_train, y_train, batch_size=32, epochs=5, callbacks=[tensorboard])