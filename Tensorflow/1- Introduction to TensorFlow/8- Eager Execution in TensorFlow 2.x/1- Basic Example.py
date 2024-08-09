import tensorflow as tf

# Eager execution is enabled by default in TensorFlow 2.x
print(tf.executing_eagerly())

# Simple operations
a = tf.constant(2.0)
b = tf.constant(3.0)
c = a + b
print("Sum:", c)
