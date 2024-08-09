import tensorflow as tf
from sklearn.metrics import accuracy_score

# Assuming y_true and y_pred are numpy arrays or tensors
y_true = [1, 0, 1, 1]
y_pred = [1, 0, 0, 1]

accuracy = accuracy_score(y_true, y_pred)
print('Accuracy:', accuracy)
