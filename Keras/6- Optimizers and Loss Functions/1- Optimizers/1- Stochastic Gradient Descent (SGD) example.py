from tensorflow.keras.optimizers import SGD

# Using default parameters
optimizer = SGD()

# Custom parameters
optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
