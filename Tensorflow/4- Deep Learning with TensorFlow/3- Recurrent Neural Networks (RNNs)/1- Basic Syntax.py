import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Create a Recurrent Neural Network
model = Sequential([
    SimpleRNN(50, activation='relu', input_shape=(timesteps, features)),
    Dense(1)  # For regression; use Dense(num_classes, activation='softmax') for classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='mean_squared_error',  # Use 'sparse_categorical_crossentropy' for classification
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
