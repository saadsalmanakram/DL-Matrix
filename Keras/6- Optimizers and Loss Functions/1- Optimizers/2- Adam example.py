from tensorflow.keras.optimizers import Adam

# Using default parameters
optimizer = Adam()

# Custom parameters
optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
