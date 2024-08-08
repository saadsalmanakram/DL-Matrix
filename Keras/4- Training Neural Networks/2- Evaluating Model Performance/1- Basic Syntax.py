# Evaluate the model
loss, accuracy = model.evaluate(
    X_test,               # Test data
    y_test,               # Test labels
    verbose=1             # Verbosity mode (0, 1, 2)
)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")
