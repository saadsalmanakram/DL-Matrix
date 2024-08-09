import tensorflow as tf
import tensorflow_datasets as tfds

# Load MNIST dataset
(train_data, test_data), ds_info = tfds.load('mnist', split=['train', 'test'], as_supervised=True, with_info=True)

# View dataset information
print(ds_info)

# Preprocess and batch the data
train_data = train_data.batch(32).shuffle(10000).prefetch(tf.data.AUTOTUNE)
test_data = test_data.batch(32).prefetch(tf.data.AUTOTUNE)

# Print a sample batch
for image, label in train_data.take(1):
    print("Image shape:", image.shape)
    print("Label:", label)
