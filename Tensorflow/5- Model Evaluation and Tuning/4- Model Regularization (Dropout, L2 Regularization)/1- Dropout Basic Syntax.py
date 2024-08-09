from tensorflow.keras.layers import Dropout

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])
