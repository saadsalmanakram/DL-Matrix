from tensorflow.keras.losses import CategoricalCrossentropy

loss_function = CategoricalCrossentropy()

# Or directly in model.compile
model.compile(optimizer='adam', loss='categorical_crossentropy')
