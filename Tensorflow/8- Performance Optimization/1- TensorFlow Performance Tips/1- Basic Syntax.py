import tensorflow as tf

# Enable mixed precision training for better performance
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Use tf.function to compile Python functions into TensorFlow graphs
@tf.function
def optimized_function(x):
    return x * 2

# Example usage
x = tf.constant([1.0, 2.0, 3.0])
print(optimized_function(x))

# Use `tf.data.Dataset` for efficient data loading
def preprocess(image, label):
    return image / 255.0, label

dataset = tf.data.Dataset.from_tensor_slices((images, labels))
dataset = dataset.map(preprocess).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

# Disable eager execution for better performance (in TensorFlow 1.x)
tf.compat.v1.disable_eager_execution()
