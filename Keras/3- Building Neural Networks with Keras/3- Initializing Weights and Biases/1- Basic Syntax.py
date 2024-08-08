from keras.initializers import GlorotUniform, RandomNormal

# Using Glorot uniform initializer (also known as Xavier initializer)
model.add(Dense(units=64, kernel_initializer=GlorotUniform(), input_shape=(784,)))

# Using Random Normal initializer for biases
model.add(Dense(units=64, bias_initializer=RandomNormal(mean=0.0, std=0.05)))
