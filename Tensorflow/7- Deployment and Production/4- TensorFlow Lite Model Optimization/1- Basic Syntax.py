import tensorflow as tf

# Convert a model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_saved_model('path/to/saved_model_directory')
tflite_model = converter.convert()

# Save the model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# For Quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

# For Pruning (during training)
import tensorflow_model_optimization as tfmot

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
model = prune_low_magnitude(model)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
