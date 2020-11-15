import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random



# linear Regression
# - linear regression indicates the data's character well
population_growth = [0.3, -0.78, 1.26, 0.03, 1,11, 15.17, 0.24, -0.24, -0.47, -0.77, -0.37,
                    -0.85, -0.41, -0.27, -0.76, 2.66]

population_elder = [12.27, 14.44, 11.87, 18.75, 17.52, 9.29, 16.37, 19.78, 19.51, 12.65, 14.75,
                   10.72, 21.94, 12.83, 15.51, 17.14, 14.42]

plt.plot(population_growth, population_elder, "bo")
plt.xlabel("population_growth")
plt.ylabel("population_elder")
plt.show()

X = [0.3, -0.78, 1.26, 0.03, 1,11, 15.17, 0.24, -0.24, -0.47, -0.77, -0.37,
                    -0.85, -0.41, -0.27, -0.76, 2.66]

Y = [12.27, 14.44, 11.87, 18.75, 17.52, 9.29, 16.37, 19.78, 19.51, 12.65, 14.75,
                   10.72, 21.94, 12.83, 15.51, 17.14, 14.42]

# Generate single Random number
print(random.random())

# init w and b by Random number
# 이렇게해줌으로서 random 변수를 선언할 수 있다. 
w = tf.Variable(random.random())
b = tf.Variable(random.random())


# return loss value
def return_loss():
    #dot Opeertaion
    y_pred = w*X + b
    
    #Mean Square Error
    loss = tf.reduce_mean((y_pred - Y)**2)
    return loss

# set optimizer. use adam
optimizer = tf.optimizers.Adam(lr=0.07)

for i in range(1000):
  optimizer.minimize(return_loss, var_list=[w, b])

  if i % 100 == 99:
    print(i, 'w:', w.numpy() ,'b:', b.numpy(), 'loss: ', return_loss().numpy())

line_x = np.arange(min(X),max(X),0.01)
line_y = w*line_x + b

plt.plot(line_x, line_y, 'g-')
plt.plot(X, Y, 'bo')
plt.xlabel("population_growth Rate (%)")
plt.ylabel("population_elder Rate (%)")
plt.show()

# Polynomial regression
X = [0.3, -0.78, 1.26, 0.03, 1,11, 15.17, 0.24, -0.24, -0.47, -0.77, -0.37,
                    -0.85, -0.41, -0.27, -0.76, 2.66]

Y = [12.27, 14.44, 11.87, 18.75, 17.52, 9.29, 16.37, 19.78, 19.51, 12.65, 14.75,
                   10.72, 21.94, 12.83, 15.51, 17.14, 14.42]

# generate single Random number
print(random.random())

# initial w1, w2 and by Random nubmer
w1 = tf.Variable(random.random())
w2 = tf.Variable(random.random())
b = tf.Variable(random.random())

# return loss value
def return_loss():

  #dot Operation
  y_pred = w1 * X * X + w2 * X + b

  # Mean Sqaure Error
  loss = tf.reduce_mean((y_pred - Y)**2)
  return loss

optimizer = tf.optimizers.Adam(lr=0.07)

for i in range(1000):
  optimizer.minimize(return_loss, var_list = [w1, w2, b])

  if i % 100 == 99:
    print(i, 'w : ', w.numpy(), ' b : ', b.numpy(), ' loss : ', return_loss().numpy())

# draw the Regression Line
line_x = np.arange(min(X), max(X), 0.01)
line_y = w1*line_x*line_x + w2*line_x + b

plt.plot(line_x, line_y, 'g-')
plt.plot(X, Y, 'bo')
plt.xlabel("population_growth Rate (%)")
plt.ylabel("population_elder Rate (%)")
plt.show()