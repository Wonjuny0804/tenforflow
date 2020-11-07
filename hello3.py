import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = np.array([[1,1],[1, 0],[0,1],[0, 0]])
y = np.array([[0],[1],[1],[0]])

# making model 모델 만들기
model = tf.keras.Sequential([
                            # 노드 두개  activation function=sigmoid, input_shpae
    tf.keras.layers.Dense(units=3, activation="sigmoid", input_shape=(2,)), # 첫번째 단
    tf.keras.layers.Dense(units=1, activation="sigmoid"), # 두번째 단
]) 

# Setting model
# 모델 갱신, 손실함수에 대한 최적화 코드  optimizer, loss-funtion
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1), loss='mse')

# check and print model
print(model.summary())


# epoch와 batch size를 조정
# why this history
history = model.fit(x, y, epochs=1000, batch_size = 1)

print("===================================================")
print(model.predict(x))
# when rounded
print("===================================================")
print(tf.math.round(model.predict(x)))

plt.plot(history.history['loss'])
plt.show()