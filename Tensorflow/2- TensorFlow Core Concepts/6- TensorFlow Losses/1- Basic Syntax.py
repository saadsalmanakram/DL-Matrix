import tensorflow as tf

# True and predicted labels
y_true = tf.constant([0., 1., 2.])
y_pred = tf.constant([0.5, 0.6, 0.4])

# Mean Squared Error Loss
mse_loss = tf.keras.losses.MeanSquaredError()
loss = mse_loss(y_true, y_pred)
print("MSE Loss:", loss.numpy())

# Categorical Cross-Entropy Loss
y_true_cat = tf.constant([[0., 1., 0.]])
y_pred_cat = tf.constant([[0.2, 0.8, 0.0]])
cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
loss_cat = cross_entropy_loss(y_true_cat, y_pred_cat)
print("Categorical Cross-Entropy Loss:", loss_cat.numpy())
