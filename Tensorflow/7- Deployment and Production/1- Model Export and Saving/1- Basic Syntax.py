import tensorflow as tf

# Assuming `model` is your trained model
# Saving the model in the SavedModel format (recommended)
model.save('pathtosave_model_directory')

# Saving the model in HDF5 format
model.save('pathtosave_model.h5')

# To load the model back
loaded_model = tf.keras.models.load_model('pathtosave_model_directory')
# or
loaded_model = tf.keras.models.load_model('pathtosave_model.h5')
