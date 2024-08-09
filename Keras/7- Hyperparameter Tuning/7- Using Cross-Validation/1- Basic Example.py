from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier

# Function to create model
def create_model():
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Wrap Keras model in KerasClassifier
model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)

# Apply cross-validation
kfold = cross_val_score(model, X, Y, cv=5)
print(f"Cross-Validation Accuracy: {kfold.mean()} (+/- {kfold.std()})")
