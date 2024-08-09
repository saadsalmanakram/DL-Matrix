from tensorflow.keras.layers import Dropout

model = Sequential([
    Dense(128, activation='relu', input_shape=(input_dim,)),
    Dropout(0.5),  # Dropout rate of 50%
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(output_dim, activation='softmax')
])
