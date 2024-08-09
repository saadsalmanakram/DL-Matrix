import tensorflow as tf

# TensorFlow Constants
const = tf.constant([1.0, 2.0], name="const")
print("Constant:", const)

# TensorFlow Variables
var = tf.Variable([1.0, 2.0], name="var")
print("Variable before update:", var)

# Assign new values to the variable
var.assign([3.0, 4.0])
print("Variable after update:", var)
