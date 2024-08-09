import tensorflow_hub as hub

# Load a pre-trained model from TensorFlow Hub
model = hub.KerasLayer('httpstfhub.devgoogleimagenetresnet_v2_50feature_vector4', input_shape=(224, 224, 3))

# Use the model in your pipeline
model.trainable = False
model = tf.keras.Sequential([
    model,
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
