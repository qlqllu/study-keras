import numpy as np
from tensorflow import keras
from keract import get_activations, display_activations


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

model = keras.models.load_model('./modules/mnist')

keract_inputs = x_test[:1]
activations = get_activations(model, keract_inputs)
display_activations(activations, cmap="gray", save=True, directory="./layer_images")





