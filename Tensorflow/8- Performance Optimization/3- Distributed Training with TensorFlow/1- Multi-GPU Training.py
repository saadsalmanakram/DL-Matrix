import tensorflow as tf

# Create a mirrored strategy for multi-GPU training
strategy = tf.distribute.MirroredStrategy()

# Use the strategy in the model training process
with strategy.scope():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_dataset, epochs=5)
