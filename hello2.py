import tensorflow as tf
import numpy as np
import math

def sigmoid(x) :
    return 1 / (1+math.exp(-x))

# initialize (AND)
x = np.array([[1,1],[1, 0],[0,1],[0, 0]])
y = np.array([[1],[0],[0],[0]])

weight_ = tf.random.normal([2], 0, 1)
bias_ = tf.random.normal([1], 0, 1)

for i in range(3001):
    error_sum = 0

    for j in range(len(x)):
        output = sigmoid(np.sum(x[j]*weight_) + bias_) # 시그모이드함수 output 값 weight를 input과 곱해줌
        error_ = y[j][0] - output # 우리가 노리는 값 - 실제로 나온 값 그것의 격차
        weight_ = weight_ + 0.1*x[j]*error_ # 가중치 값을 구한다. 
        bias_ = bias_ +  0.1 * error_

        # total error 
        error_sum += error_

    if (i%300 == 0):
        print(i,":", round(error_sum, 6))
for i in range(len(x)):
    print('X:', x[i], 'Y:', y[i], 'Output:', round(sigmoid(np.sum(x[i]*weight_)+bias_), 1))

# initialize (AND)
x = np.array([[1,1],[1, 0],[0,1],[0, 0]])
y = np.array([[1],[1],[1],[0]])

weight_ = tf.random.normal([2], 0, 1)
bias_ = tf.random.normal([1], 0, 1)

for i in range(3001):
    error_sum = 0

    for j in range(len(x)):
        output = sigmoid(np.sum(x[j]*weight_) + bias_) # 시그모이드함수 output 값 weight를 input과 곱해줌
        error_ = y[j][0] - output # 우리가 노리는 값 - 실제로 나온 값 그것의 격차
        weight_ = weight_ + 0.1*x[j]*error_ # 가중치 값을 구한다. 
        bias_ = bias_ +  0.1 * error_

        # total error 
        error_sum += error_

    if (i%300 == 0):
        print(i,":", round(error_sum, 6))
for i in range(len(x)):
    print('X:', x[i], 'Y:', y[i], 'Output:', round(sigmoid(np.sum(x[i]*weight_)+bias_), 1))

#### XOR 문제 해결하기
x = np.array([[1,1],[1, 0],[0,1],[0, 0]])
y = np.array([[0],[1],[1],[0]])

weight_ = tf.random.normal([2], 0, 1)
bias_ = tf.random.normal([1], 0, 1)

for i in range(3001):
    error_sum = 0

    for j in range(len(x)):
        output = sigmoid(np.sum(x[j]*weight_) + bias_) # 시그모이드함수 output 값 weight를 input과 곱해줌
        error_ = y[j][0] - output # 우리가 노리는 값 - 실제로 나온 값 그것의 격차
        weight_ = weight_ + 0.1*x[j]*error_ # 가중치 값을 구한다. 
        bias_ = bias_ +  0.1 * error_

        # total error 
        error_sum += error_

    if (i%300 == 0):
        print(i,":", round(error_sum, 6))
for i in range(len(x)):
    print('X:', x[i], 'Y:', y[i], 'Output:', round(sigmoid(np.sum(x[i]*weight_)+bias_), 1))