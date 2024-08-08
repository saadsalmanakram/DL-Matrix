from keras.layers import Input, Dense
from keras.models import Model

inputs = Input(shape=(32,))
x = Dense(64, activation='relu')(inputs)
outputs = Dense(10, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)
