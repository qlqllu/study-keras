import numpy as np
from tensorflow import keras
from tensorflow.python.keras import layers
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


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

"""
## Build the model
"""

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", name="conv1"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool1"),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", name="conv2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool2"),
        layers.Flatten(name="flat"),
        layers.Dropout(0.5, name="dropout"),
        layers.Dense(num_classes, activation="softmax", name="dense"),
    ]
)

model.summary()

"""
## Train the model
"""

batch_size = 128
epochs = 1

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
score = model.evaluate(x_test, y_test, verbose=0)

"""
## Evaluate the trained model
"""  
print("Test loss:", score[0])
print("Test accuracy:", score[1])

keract_inputs = x_test[:1]
keract_targets = y_test[:1]
activations = get_activations(model, keract_inputs)
display_activations(activations, cmap="gray", save=True, directory="./layer_images")





