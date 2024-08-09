from tensorflow.keras.regularizers import l1, l2, l1_l2

model = Sequential([
    Dense(128, activation='relu', input_shape=(input_dim,),
          kernel_regularizer=l1(0.01)),  # L1 regularization with lambda=0.01
    Dense(64, activation='relu',
          kernel_regularizer=l2(0.01)),  # L2 regularization with lambda=0.01
    Dense(32, activation='relu',
          kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),  # Combined L1 and L2 regularization
    Dense(output_dim, activation='softmax')
])
