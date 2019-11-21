class Models:
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

def get_model():
    return M.model
