# Import necessary libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Define the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(input_dim,)),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
history = model.fit(
    X_train,              # Training data
    y_train,              # Training labels
    epochs=20,            # Number of epochs
    batch_size=32,        # Batch size
    validation_split=0.2, # Fraction of data to use as validation set
    verbose=1             # Verbosity mode (0, 1, 2)
)
