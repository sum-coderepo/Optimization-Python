import tensorflow as tf
mnist = tf.keras.datasets.mnist
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.tanh),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1)

test_predictions = model.predict_classes(x_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true = y_test, y_pred = test_predictions)
print(cm)


