import tensorflow as tf

# Load a pre-trained Keras model
model = tf.keras.models.load_model('path/to/model')

# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
