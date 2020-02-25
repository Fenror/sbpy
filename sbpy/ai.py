""" This module contains various machine learning based functions. """

from tensorflow import keras
import numpy as np


Nc = 21
Nf = 161

model = keras.models.Sequential()

model.add(keras.layers.Dense(16, input_shape=(1,)))
model.add(keras.layers.Dense(16))
model.add(keras.layers.Dense(16))
model.add(keras.layers.Dense(16))
model.add(keras.layers.Dense(16))
model.add(keras.layers.Dense(16))
model.add(keras.layers.Dense(Nc*4*4, activation='linear'))

print(model(np.array([[1],[2]])))
