#BostionHosting dataset is in tensorflow
import tensorflow as tf
from tensorflow.keras.datasets import boston_housing as BH

# boston_housing dataset Load 
(train_X, train_Y), (test_X,test_Y) = BH.load_data()

#number of boston_housing dataset
print("Train_X : \t", len(train_X), ", Train_Y : \t", len(train_Y))
print("Test_X : \t", len(test_X), ", Test_Y : \t", len(test_Y))
print()
print("The Form of Train_X Data: \n", train_X[0])
print("The Form of Train_Y Data: \n", train_Y[0])

#Data Standardization

# get mean and std
x_mean = train_X.mean(axis=0)
x_std = train_X.std(axis=0)

y_mean = train_Y.mean(axis=0)
y_std = train_Y.std(axis=0)

#Do Standardization Formula
train_X -= x_mean
train_X /= x_std
test_X -= x_mean
test_X /= x_std


train_Y -= y_mean
train_Y /= y_std
test_Y -= y_mean
test_Y /= y_std

print("The form of Standardization Train_X Data: \n", train_X[0])
print("The form of Standardization Train_Y Data: ", train_Y[0])

model = tf.keras.Sequential([
  tf.keras.layers.Dense(units=52, activatiuon='relu', input_shoe=(13,)),
  tf.keras.layers.Dense(units=39, activatiuon='relu'),
  tf.keras.layers.Dense(units=26, activatiuon='relu'),
  tf.keras.layers.Dense(units=1)
])
# set model and model summary
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.07), loss='mse')
model.summary()

#v Learning model
history = model.fit(train_X, train_Y, epochs=25, batch_size=32, validation_split=0.25,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3,monitor='val_loss')])
print(history)

plt.plot(history.history['loss'], 'b--', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.show()

# evaluate model using test data
testLoss = model.evaluate(test_X, test_Y)
print('\nTest Loss : ', testLoss)

# compare with Real and Prediction 
pred_Y = model.predict(test_X)

print(pred_Y)

plt.figure(figsuze=(5,5))
plt.plot(test_Y, pred_Y, 'b.')
plt.axis([min(test_Y), max(test_Y), min(test_Y), max(test_Y)])

plt.plot([min(test_Y), max(test_Y)],[min(test_Y), max(test_Y)], ls='-.', c='.5')

plt.xlabel('test_Y')
plt.ylabel('pred_Y')

plt.show()