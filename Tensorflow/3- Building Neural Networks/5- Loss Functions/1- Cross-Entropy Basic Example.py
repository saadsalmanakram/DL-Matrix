import tensorflow as tf

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, activation='sigmoid', input_shape=[1])
])

# Compile the model with Cross-Entropy loss
model.compile(optimizer='adam', loss='binary_crossentropy')
