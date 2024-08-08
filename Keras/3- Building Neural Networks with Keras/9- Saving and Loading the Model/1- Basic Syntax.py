# Saving the model
model.save('my_model.h5')

# Loading the model
from keras.models import load_model
loaded_model = load_model('my_model.h5')
