from keras.models import Sequential
from keras.layers import Dense, Dropout

# Simple model to demonstrate overfitting
def create_overfitting_model():
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(10000,)))
    model.add(Dropout(0.5))  # Adding dropout can reduce overfitting
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Training with a small dataset can cause overfitting
model = create_overfitting_model()
history = model.fit(X_train, y_train, epochs=20, batch_size=512, validation_data=(X_val, y_val))

# Plotting training and validation loss to visualize overfitting
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
