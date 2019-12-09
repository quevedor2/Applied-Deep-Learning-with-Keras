class BinaryModels:
    def __init__(self, X, model=0):
        self.X = X
        self.model=model
    
    def log_model(self):
        lr_model = Sequential()
        lr_model.add(Dense(1, input_shape=(X.shape[1],), activation='sigmoid'))
        
        lr_model.compile(Adam(lr=0.01), 'binary_crossentropy', metrics=['accuracy'])
        self.model=lr_model
    
    def ann_model(self):
        deep_model = Sequential()
        deep_model.add(Dense(64, input_shape=(X.shape[1],), activation='tanh'))
        deep_model.add(Dense(16, activation='tanh'))
        deep_model.add(Dense(1, activation='sigmoid'))
        
        deep_model.compile(Adam(lr=0.01), 'binary_crossentropy', metrics=['accuracy'])
        self.model=deep_model
    
    def ann_vis_model(self):
        deep_model = Sequential()
        deep_model.add(Dense(64, input_shape=(X.shape[1],), activation='tanh'))
        deep_model.add(Dense(16, activation='tanh'))
        deep_model.add(Dense(2, activation='tanh'))
        deep_model.add(Dense(1, activation='sigmoid'))
        
        deep_model.compile(Adam(lr=0.01), 'binary_crossentropy', metrics=['accuracy'])
        self.model=deep_model

class MulticlassModels:
    def __init__(self, X, model=0):
        self.X = X
        self.model=model
    
    def sr_model(self):
        print("Softmax-regression model")
        sr_model = Sequential()
        sr_model.add(Dense(3, input_shape=(X.shape[1],), activation='softmax'))
        
        sr_model.compile(Adam(lr=0.1), loss='categorical_crossentropy', metrics=['accuracy'])
        self.model=sr_model
    
    def ann_model(self):
        print("ANN model")
        deep_model = Sequential()
        deep_model.add(Dense(32, input_shape=(X.shape[1],), activation='relu'))
        deep_model.add(Dense(16, input_shape=(X.shape[1],), activation='relu'))
        deep_model.add(Dense(3, activation='softmax'))
        
        deep_model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
        self.model=deep_model

class RegressionModels:
    def __init__(self, X, model=0):
        self.X = X
        self.model=model
    
    def linear_model(self):
        print("Linear regression model")
        linr_model = Sequential()
        linr_model.add(Dense(1, input_shape=(X.shape[1],)))
        
        linr_model.compile('adam', 'mean_squared_error')
        self.model=linr_model
    
    def ann_model(self):
        print("ANN model")
        deep_model = Sequential()
        deep_model.add(Dense(32, input_shape=(X.shape[1],), activation='relu'))
        deep_model.add(Dense(16, activation='relu'))
        deep_model.add(Dense(8, activation='relu'))
        deep_model.add(Dense(1))
        
        deep_model.compile('adam', 'mean_squared_error')
        self.model=deep_model

class CNN:
    def __init__(self, X, model=0):
        self.X = X
        self.model=model
    
    def CNN(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_1',
                         input_shape=(150, 150, 3)))
        model.add(MaxPooling2D((2, 2), name='maxpool_1'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_2'))
        model.add(MaxPooling2D((2, 2), name='maxpool_2'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_3'))
        model.add(MaxPooling2D((2, 2), name='maxpool_3'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_4'))
        model.add(MaxPooling2D((2, 2), name='maxpool_4'))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu', name='dense_1'))
        model.add(Dense(256, activation='relu', name='dense_2'))
        model.add(Dense(1, activation='sigmoid', name='output'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model=model

def get_model():
    return M.model
