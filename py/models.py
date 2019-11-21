def log_model(X):
    lr_model = Sequential()
    lr_model.add(Dense(1, input_shape=(X.shape[1],), activation='sigmoid'))
    lr_model.compile(Adam(lr=0.01), 'binary_crossentropy', metrics=['accuracy'])
    return lr_model
