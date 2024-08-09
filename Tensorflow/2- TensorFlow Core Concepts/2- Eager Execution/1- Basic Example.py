import tensorflow as tf

# Enable eager execution
tf.config.run_functions_eagerly(True)

# Now operations execute immediately
a = tf.constant(5)
b = tf.constant(3)
c = a + b
print("Result of a + b:", c.numpy())
