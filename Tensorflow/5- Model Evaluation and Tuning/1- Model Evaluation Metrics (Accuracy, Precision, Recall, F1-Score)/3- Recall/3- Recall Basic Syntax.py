from sklearn.metrics import recall_score

recall = recall_score(y_true, y_pred)
print('Recall:', recall)
