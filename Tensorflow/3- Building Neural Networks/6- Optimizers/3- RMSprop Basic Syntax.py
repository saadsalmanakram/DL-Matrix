import tensorflow as tf

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Compile the model with RMSprop optimizer
model.compile(optimizer='rmsprop', loss='mean_squared_error')
