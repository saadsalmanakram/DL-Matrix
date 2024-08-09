import tensorflow_hub as hub
import tensorflow as tf

model = hub.load("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4")
