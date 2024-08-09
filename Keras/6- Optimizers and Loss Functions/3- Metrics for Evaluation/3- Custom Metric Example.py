def custom_metric(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)))

# Using the custom metric
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[custom_metric])
