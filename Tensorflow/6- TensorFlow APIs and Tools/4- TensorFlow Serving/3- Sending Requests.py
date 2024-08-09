import requests
import json

data = json.dumps({"signature_name": "serving_default", "instances": [{"input": x.tolist()}]})
headers = {"Content-Type": "application/json"}
response = requests.post("http://localhost:8501/v1/models/model_name:predict", data=data, headers=headers)
predictions = response.json()
