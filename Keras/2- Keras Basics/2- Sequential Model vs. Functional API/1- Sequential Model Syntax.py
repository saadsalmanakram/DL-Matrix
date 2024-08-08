from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(32,)))
model.add(Dense(10, activation='softmax'))
