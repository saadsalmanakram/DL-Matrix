import tensorflow as tf

# Create a dataset from a list of values
dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])

# Define a simple pipeline
dataset = dataset.map(lambda x: x * 2)  # Multiply each element by 2
dataset = dataset.batch(2)  # Batch the data

# Iterate over the dataset
for batch in dataset:
    print(batch.numpy())
