import tensorflow as tf
from tensorflow.python.profiler import profiler_v2 as profiler

# Enable profiling
profiler.start('logdir')

# Model training code
model.fit(train_dataset, epochs=5)

# Stop profiling
profiler.stop()

# Visualize results in TensorBoard
# Use the command: `tensorboard --logdir=logdir` to view the profiler data
