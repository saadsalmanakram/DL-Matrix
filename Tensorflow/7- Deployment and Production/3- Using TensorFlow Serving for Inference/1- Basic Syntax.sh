# Install TensorFlow Serving
pip install tensorflow-serving-api

# Start TensorFlow Serving with your model
tensorflow_model_server --rest_api_port=8501 --model_name=my_model --model_base_path="/path/to/saved_model_directory"

# Make a prediction using curl
curl -d '{"instances": [[1.0, 2.0, 5.0]]}' -X POST http://localhost:8501/v1/models/my_model:predict
