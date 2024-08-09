from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

model.fit(datagen.flow(X_train, y_train, batch_size=32),
          validation_data=(X_val, y_val), epochs=100)
