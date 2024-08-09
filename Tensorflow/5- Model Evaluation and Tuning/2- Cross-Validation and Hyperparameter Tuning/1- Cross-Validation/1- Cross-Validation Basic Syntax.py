from sklearn.model_selection import cross_val_score
import numpy as np

# Example with a scikit-learn model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation
print('Cross-Validation Scores:', scores)
print('Mean Cross-Validation Score:', np.mean(scores))
