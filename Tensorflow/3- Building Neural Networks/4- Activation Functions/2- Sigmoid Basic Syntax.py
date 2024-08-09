import tensorflow as tf

# Define the model with Sigmoid activation
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, activation='sigmoid', input_shape=[1])
])
