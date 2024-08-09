from tensorflow.keras.regularizers import l2

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    tf.keras.layers.Dense(10, activation='softmax')
])
