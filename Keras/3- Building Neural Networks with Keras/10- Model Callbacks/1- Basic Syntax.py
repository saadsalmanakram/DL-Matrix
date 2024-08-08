from keras.callbacks import EarlyStopping, ModelCheckpoint

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Model checkpoint
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

# Training with callbacks
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2,
          callbacks=[early_stopping, checkpoint])
