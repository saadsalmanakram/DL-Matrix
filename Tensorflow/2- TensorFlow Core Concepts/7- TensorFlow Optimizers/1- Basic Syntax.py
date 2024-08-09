import tensorflow as tf

# Simple model with a variable
W = tf.Variable(5.0, name="weight")

# Define a simple loss function
def loss_fn():
    return (W - 3.0) ** 2

# Choose an optimizer
optimizer = tf.optimizers.SGD(learning_rate=0.01)

# Apply the optimizer
for step in range(100):
    optimizer.minimize(loss_fn, var_list=[W])

print("Optimized Weight:", W.numpy())
