import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

fig, axes = plt.subplots(10, 10, figsize=(8,8))

for i,ax in enumerate(axes.flat):
  ax.imshow(x_test[i])

plt.show()
