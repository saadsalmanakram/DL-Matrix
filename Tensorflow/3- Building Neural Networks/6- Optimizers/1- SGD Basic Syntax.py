import tensorflow as tf

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Compile the model with SGD optimizer
model.compile(optimizer='sgd', loss='mean_squared_error')
