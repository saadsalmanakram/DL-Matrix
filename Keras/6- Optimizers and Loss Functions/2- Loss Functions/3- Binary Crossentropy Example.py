from tensorflow.keras.losses import BinaryCrossentropy

loss_function = BinaryCrossentropy()

# Or directly in model.compile
model.compile(optimizer='adam', loss='binary_crossentropy')
