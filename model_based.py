import tensorflow.keras as keras
import numpy as np

class BaseModel:
    def __init__(self):
        pass
    
    def _build_model(self,obs_space,out_space,n_layers=3,n_units=128,activations='tanh'):
        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(n_units,activations,input_shape=[obs_space]))
        for _ in range(n_layers-1):
            self.model.add(keras.layers.Dense(n_units,activations))
        self.model.add(keras.layers.Dense(out_space,activation='tanh'))

    def train(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

class dynamics_model(BaseModel):
    
    def __init__(self,in_shape,out_shape,activations='tanh',\
                 n_layers=3,n_units=128,\
                 learn_rate=0.001):
        super().__init__()
        self.n_layers = n_layers
        self.n_units = n_units
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.activations = activations
        self._build_model(self.in_shape,self.out_shape,self.n_layers,\
                          self.n_units,self.activations)
        optimizer = keras.optimizers.Adam(learn_rate)
        self.model.compile(optimizer,loss='mse',metrics=['MeanSquaredError'])

    def train(self, train_x, train_y, epochs=30):
        self.history = self.model.fit(train_x,train_y,batch_size=64,epochs=epochs,\
                                 verbose=2,validation_split=0.2)
        
    def predict(self, test_x):
        return self.model.predict(test_x)

class q_model(BaseModel):

    def __init__(self,in_shape,out_shape,activations='tanh',\
                 n_layers=3,n_units=128,\
                 learn_rate=0.001):
        super().__init__()
        self.n_layers = n_layers
        self.n_units = n_units
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.activations = activations
        self._build_model(self.in_shape,self.out_shape,self.n_layers,\
                          self.n_units,self.activations)
        optimizer = keras.optimizers.Adam(learn_rate)
        self.model.compile(optimizer,loss='mse',metrics=['MeanSquaredError'])

    def train(self, train_x, train_y, epochs=15):
        self.history = self.model.fit(train_x,train_y,batch_size=64,epochs=epochs,\
                                 verbose=2,validation_split=0.2)
        
    def predict(self, test_x):
        return self.model.predict(test_x)
