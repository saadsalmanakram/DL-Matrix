from tensorflow.keras.callbacks import TensorBoard

# Define TensorBoard callback
tensorboard = TensorBoard(
    log_dir='logs',           # Directory where logs will be saved
    histogram_freq=1,         # Frequency (in epochs) at which to compute activation and weight histograms
    write_graph=True,         # Whether to visualize the graph
    write_images=True,        # Whether to write model weights to visualize as images
    update_freq='epoch'       # Frequency (in steps) at which to write logs
)

# Fit the model with TensorBoard
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[tensorboard]
)
