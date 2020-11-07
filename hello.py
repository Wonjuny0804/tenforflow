import tensorflow as tf
import numpy as np
import math

def sigmoid(x) :
    return 1 / (1+math.exp(-x))



# 랜덤넘버 생성 방식
# 랜덤 분포를 만드는 방식 균일분포
rand = tf.random.uniform([3], 0, 1) # 0에서 1사이 값 3개 랜덤
print(rand)
print("\n\n")

rand = tf.random.uniform([2,2], 7, 6) # 2x2 형태로 6-7 사이 값 랜덤 추출
print(rand)

# 2. Normal Distribution
rand= tf.random.normal([3], 0, 1) # 평균값0, 표준편차
print(rand, end="\n\n")
# 평균값 7, 표준쳔차 0.01인 값추출
rand = tf.random.normal([2,2], 7, 0.01) 
print(rand)

for i in range(-7,7):
    output = sigmoid(i)
    print(i,":",round(output, 5))



#### learning gradient descent

##### 2-1. Without bias, Learning weight
inputValue = 0.1
target = 1
weight = tf.random.normal([1], 0, 1) # tf가 알아서 초기화해줌
bias = tf.random.normal([1], 0 , 1)

print("Repetition \t Error \t Out put")

for i in range(10000):
    output = sigmoid(inputValue*weight + bias) # 시그모이드함수 output 값 weight를 input과 곱해줌
    error_ = (target - output)**2  # 우리가 노리는 값 - 실제로 나온 값 그것의 격차
    weight = weight + 0.1*inputValue*error_ # 가중치 값을 구한다. 
    bias = bias + 1 * 0.1 * error_


    if (i % 1000 == 0):
        print(i, '\t', round(error_, 6), '\t', round(output, 6))