import tensorflow as tf

# Creating a Tensor
tensor = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
print("Tensor:", tensor)

# Basic Tensor Operations
result = tf.add(tensor, 2)
print("Added Tensor:", result)

# Matrix multiplication
matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2],[2]])
product = tf.matmul(matrix1, matrix2)
print("Matrix multiplication:", product)
