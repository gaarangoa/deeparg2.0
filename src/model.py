import tensorflow as tf
from tensorflow import keras


model = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(),
    keras.layers.Dense()
])
