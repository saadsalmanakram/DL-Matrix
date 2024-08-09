import tensorflow as tf

# Create a variable
W = tf.Variable([0.5, 1.0], name="weight")

# Assign a new value to the variable
W.assign([1.5, 2.0])

# Use the variable in a computation
result = W * 2
print("Result:", result.numpy())
