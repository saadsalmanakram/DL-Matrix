from tensorflow.keras import backend as K

def custom_loss(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

# Using the custom loss function
model.compile(optimizer='adam', loss=custom_loss)
