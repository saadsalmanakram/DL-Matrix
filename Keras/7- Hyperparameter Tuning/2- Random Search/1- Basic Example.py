from sklearn.model_selection import RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from scipy.stats import uniform

# Function to create model
def create_model(optimizer='adam'):
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Wrap Keras model in KerasClassifier
model = KerasClassifier(build_fn=create_model, verbose=0)

# Define random search parameters
param_dist = {'batch_size': [10, 20, 40], 'epochs': [10, 50, 100], 'optimizer': ['SGD', 'Adam', 'RMSprop']}
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, n_jobs=-1, cv=3, random_state=42)

# Perform random search
random_result = random_search.fit(X, Y)

# Summarize results
print(f"Best: {random_result.best_score_} using {random_result.best_params_}")
