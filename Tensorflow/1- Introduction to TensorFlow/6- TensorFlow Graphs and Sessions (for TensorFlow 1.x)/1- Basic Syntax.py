import tensorflow as tf

# TensorFlow 1.x style
a = tf.placeholder(tf.int32)
b = tf.placeholder(tf.int32)
add = tf.add(a, b)

with tf.Session() as sess:
    result = sess.run(add, feed_dict={a: 1, b: 2})
    print("Result:", result)
