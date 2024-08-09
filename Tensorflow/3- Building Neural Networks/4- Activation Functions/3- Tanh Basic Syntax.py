import tensorflow as tf

# Define the model with Tanh activation
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=10, activation='tanh', input_shape=[1]),
    tf.keras.layers.Dense(units=1)
])
