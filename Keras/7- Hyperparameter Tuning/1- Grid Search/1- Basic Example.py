from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense

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

# Define the grid search parameters
param_grid = {'batch_size': [10, 20, 40], 'epochs': [10, 50, 100], 'optimizer': ['SGD', 'Adam', 'RMSprop']}
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)

# Perform grid search
grid_result = grid.fit(X, Y)

# Summarize results
print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
