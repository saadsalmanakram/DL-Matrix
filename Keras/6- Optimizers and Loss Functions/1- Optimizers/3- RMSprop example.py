from tensorflow.keras.optimizers import RMSprop

# Using default parameters
optimizer = RMSprop()

# Custom parameters
optimizer = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-07)
