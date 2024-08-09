import tensorflow as tf

# Check if GPU is available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth to avoid memory pre-allocation
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# Example of assigning operations to a specific GPU
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0]])
    b = tf.constant([[3.0, 4.0]])
    c = tf.matmul(a, b)
    print(c)
