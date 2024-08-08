from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.optimizers import Adam

# Sequential API
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(32,)))
model.add(Dense(10, activation='softmax'))

# Functional API
inputs = Input(shape=(32,))
x = Dense(64, activation='relu')(inputs)
outputs = Dense(10, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)
