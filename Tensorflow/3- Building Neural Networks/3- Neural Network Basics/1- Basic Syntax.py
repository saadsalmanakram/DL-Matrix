import tensorflow as tf
import numpy as np

# Sample data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=4, activation='relu', input_shape=[2]),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=1000)

# Predictions
print(model.predict(X))
