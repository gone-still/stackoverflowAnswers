# File        :   main.py
# Version     :   1.0.1 (Keras Custom Loss)
# Description :   Keras model saving/reloading with custom loss
#                 Answer for: https://stackoverflow.com/q/78912140/12728244

# Date:       :   Aug 28, 2024
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   Creative Commons CC0

import numpy as np
import tensorflow as tf

from keras import Sequential
from keras.layers import Input, Dense

# Model architecture:
model = Sequential()
model.add(Input(shape=(100,)))
model.add(Dense(units=5, activation='relu'))
model.add(Dense(units=1))

# Custom Loss:
def custom_loss(y_true, y_pred):
    squared_difference = tf.math.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)  

# Compile model:
model.compile(optimizer="adam", loss=custom_loss, metrics= ["mean_squared_error"])

# Show summary:
model.summary()

# Some random inputs/targets:
x=np.random.rand(300,100)
y=np.random.rand(300,2)

# Fit the model for 5 epochs:
model.fit(x,y,batch_size=100, epochs=5)

# Save model:
path = 'saved_model/myModel.keras'

# Explicit deletion of the model object:
model.save(path)
print("Model saved")

# Load model:
del model
model = tf.keras.models.load_model(path, compile=False, custom_objects={"custom_loss": custom_loss})
print("Model reloaded")

# Compile:
model.compile(optimizer="adam", loss=custom_loss, metrics= ["mean_squared_error"])

# Continue training for 5 more epochs:
model.fit(x, y, batch_size=100, epochs=5)
print("Done fitting")