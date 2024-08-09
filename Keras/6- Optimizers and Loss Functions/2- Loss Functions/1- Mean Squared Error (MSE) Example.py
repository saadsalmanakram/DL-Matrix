from tensorflow.keras.losses import MeanSquaredError

loss_function = MeanSquaredError()

# Or directly in model.compile
model.compile(optimizer='adam', loss='mse')
