import tensorflow as tf
from random import randint
import matplotlib.pyplot as plt
import tensorflow_hub as hub


#build our nn in this file (from mnist.py)
def build_model(loaded):
  x = tf.keras.layers.Input(shape=(28, 28, 1), name='input_x')
  keras_layer = hub.KerasLayer(loaded, trainable=True)(x)
  model = tf.keras.Model(x, keras_layer)
  return model

#load the model (from mnist.py)
saved_model = tf.saved_model.load('model_1')
model = build_model(saved_model)

#load the dataset again for test
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#reshape again
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

#take a random image from the test dataset (10000 samples)
image_index = randint(0, 9999)
plt.imshow(x_test[image_index].reshape(28, 28), cmap='Greys')
plt.show()
#plot the number
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
#predict the number
print(f'The number is {pred.argmax()}')
