import tensorflow as tf
import numpy as np

# Sample data
X = np.array([[1], [2], [3], [4]], dtype=float)
y = np.array([[2], [3], [4], [5]], dtype=float)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=500)

# Predictions
print(model.predict([5.0]))
