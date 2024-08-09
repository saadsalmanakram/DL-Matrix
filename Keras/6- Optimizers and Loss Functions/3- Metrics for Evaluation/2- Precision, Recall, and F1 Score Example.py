from tensorflow.keras.metrics import Precision, Recall

# Adding Precision and Recall as metrics
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=[Precision(), Recall()])
