# Install TensorFlow Serving
pip install tensorflow-serving-api

# Export a trained model for serving
import tensorflow as tf

model = tf.keras.models.load_model('path_to_your_model')
tf.saved_model.save(model, 'path_to_saved_model')

# Start TensorFlow Serving
# Example command to serve the model
tensorflow_model_server --rest_api_port=8501 --model_name=my_model --model_base_path=/path_to_saved_model/

# Client-side code to make predictions
import requests
import json

data = json.dumps({"signature_name": "serving_default", "instances": [[1.0, 2.0, 5.0]]})
headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:8501/v1/models/my_model:predict', data=data, headers=headers)
predictions = json.loads(json_response.text)['predictions']
print(predictions)
