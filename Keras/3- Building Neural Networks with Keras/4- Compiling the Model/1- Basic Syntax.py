model.compile(optimizer='adam',               # Optimizer
              loss='sparse_categorical_crossentropy',  # Loss function for classification
              metrics=['accuracy'])         # Evaluation metric
