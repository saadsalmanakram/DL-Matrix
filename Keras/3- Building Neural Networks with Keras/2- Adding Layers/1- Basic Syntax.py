from keras.layers import Dense, Activation, Dropout

# Adding a Dense layer
model.add(Dense(units=64, input_shape=(784,)))  # units: number of neurons, input_shape: shape of input data

# Adding an Activation layer
model.add(Activation('relu'))  # Activation function

# Adding Dropout to prevent overfitting
model.add(Dropout(0.5))  # dropout rate: fraction of input units to drop

# Adding another Dense layer
model.add(Dense(units=10, activation='softmax'))  # final layer with softmax activation for classification
