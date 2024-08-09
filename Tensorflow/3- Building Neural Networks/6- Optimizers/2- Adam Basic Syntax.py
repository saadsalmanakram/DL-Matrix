import tensorflow as tf

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Compile the model with Adam optimizer
model.compile(optimizer='adam', loss='mean_squared_error')
