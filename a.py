import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
data = pd.read_csv('water_level_data.csv')
X = data.drop(['Water Level'], axis=1)
y = data['Water Level'].values.reshape(-1, 1)
evap = data['Evaporation'].values.reshape(-1, 1)
sanity = data['Sanity'].values.reshape(-1, 1)
perception = data['Perception'].values.reshape(-1, 1)
scaler_X = MinMaxScaler()
X = scaler_X.fit_transform(X)
scaler_y = MinMaxScaler()
y = scaler_y.fit_transform(y)
scaler_evap = MinMaxScaler()
evap = scaler_evap.fit_transform(evap)
scaler_sanity = MinMaxScaler()
sanity = scaler_sanity.fit_transform(sanity)
scaler_perception = MinMaxScaler()
perception = scaler_perception.fit_transform(perception)
X = np.concatenate((X, evap, sanity, perception), axis=1)
split = int(0.8 * len(data))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]
model = Sequential()
model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
poc /50
1/1 [==============================] - 0s 42ms/step - loss: 0.0616 - val_loss: 0.1699
Epoch 23/50
1/1 [==============================] - 0s 42ms/step - loss: 0.0588 - val_loss: 0.1702
Epoch 24/50
1/1 [==============================] - 0s 42ms/step - loss: 0.0563 - val_loss: 0.1705
Epoch 25/50
1/1 [==============================] - 0s 41ms/step - loss: 0.0539 - val_loss: 0.1703
Epoch 26/50
1/1 [==============================] - 0s 42ms/step - loss: 0.0517 - val_loss: 0.1698
Epoch 27/50
Epoch 49/50
1/1 [==============================] - 0s 43ms/step - loss: 0.0342 - val_loss: 0.1506
Epoch 50/50
1/1 [==============================] - 0s 43ms/step - loss: 0.0338 - val_loss: 0.1494
train_score = model.evaluate(X_train, y_train, verbose=0)
print('Training loss:', train_score)
Training loss: 0.033557675778865814
test_score = model.evaluate(X_test, y_test, verbose=0)
print('Testing loss:', test_score)
Testing loss: 0.1494143009185791
predictions = scaler_y.inverse_transform(model.predict(X_test))
1/1 [==============================] - 0s 73ms/step
import matplotlib.pyplot as plt
plt.plot(y_test, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.show()