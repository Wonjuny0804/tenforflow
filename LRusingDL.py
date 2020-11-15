import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

#use same data
X = [0.3, -0.78, 1.26, 0.03, 1,11, 15.17, 0.24, -0.24, -0.47, -0.77, -0.37,
                    -0.85, -0.41, -0.27, -0.76, 2.66]

Y = [12.27, 14.44, 11.87, 18.75, 17.52, 9.29, 16.37, 19.78, 19.51, 12.65, 14.75,
                   10.72, 21.94, 12.83, 15.51, 17.14, 14.42]

#Create model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units = 6, activation='tanh', input_shape=(1,)),
    tf.keras.layers.Dense(units = 1)
])

#set model
print(model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.1), loss='mse'))

# summary
print(model.summary())

history = model.fit(X, Y, epochs = 10)
print(history)

#prediction
print(model.predict(X))


# Draw Regression Line
line_x = np.arange(min(X), max(X), 0.01)
line_y = model.predict(line_x)

plt.plot(line_x, line_y, 'g-')
plt.plot(X, Y, 'bo')
plt.xlabel("population_growth Rate (%)")
plt.ylabel("population_elder Rate (%)")
plt.show()