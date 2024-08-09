from keras.callbacks import LearningRateScheduler
import math

# Learning rate schedule function
def lr_schedule(epoch):
    return 0.001 * math.exp(-0.1 * epoch)

# Define model (same as above)
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(10000,)))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Implementing learning rate scheduler
lr_scheduler = LearningRateScheduler(lr_schedule)

# Training model with learning rate scheduler
history = model.fit(X_train, y_train, epochs=20, batch_size=512, validation_data=(X_val, y_val), callbacks=[lr_scheduler])
