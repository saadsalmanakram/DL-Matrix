import tensorflow as tf

# Define cluster specification for multi-machine training
cluster_spec = {
    'worker': ['worker0:port', 'worker1:port'],
    'ps': ['ps0:port']
}

# Define a strategy for distributed training
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

# Use the strategy to train the model
with strategy.scope():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_dataset, epochs=5)
