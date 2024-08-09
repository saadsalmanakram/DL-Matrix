import tensorflow as tf

@tf.custom_gradient
def my_relu(x):
    result = tf.maximum(0.0, x)
    def grad(dy):
        # Custom gradient: 1 for positive x, 0 for negative
        return dy * tf.cast(x > 0, dtype=dy.dtype)
    return result, grad

# Use the custom gradient in a simple operation
x = tf.Variable([-1.0, 2.0, -3.0, 4.0])
with tf.GradientTape() as tape:
    y = my_relu(x)
grads = tape.gradient(y, x)

print("x:", x.numpy())
print("y:", y.numpy())
print("Gradients:", grads.numpy())
