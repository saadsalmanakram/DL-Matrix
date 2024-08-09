docker run -p 8501:8501 --name=tf_model_serving --mount type=bind,source=/path/to/model/dir,target=/models/model_name -e MODEL_NAME=model_name -t tensorflow/serving:latest
