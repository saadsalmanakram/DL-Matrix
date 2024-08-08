# Save the model after every epoch.

from tensorflow.keras.callbacks import ModelCheckpoint

# Define ModelCheckpoint callback
model_checkpoint = ModelCheckpoint(
    'model.h5',              # File path to save the model
    save_best_only=True,      # Save only the best model (based on the monitored metric)
    monitor='val_loss',      # Metric to monitor
    verbose=1                # Verbosity mode (0, 1, 2)
)

# Fit the model with ModelCheckpoint
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[model_checkpoint]
)
