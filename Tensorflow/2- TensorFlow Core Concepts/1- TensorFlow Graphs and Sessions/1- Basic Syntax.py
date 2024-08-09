import tensorflow as tf

# Define a computational graph
a = tf.constant(5, name="a")
b = tf.constant(3, name="b")
c = tf.add(a, b, name="c")

# Create a session to run the graph
with tf.compat.v1.Session() as sess:
    result = sess.run(c)
    print("Result of a + b:", result)
