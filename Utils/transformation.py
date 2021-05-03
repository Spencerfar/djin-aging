import numpy as np

class Transformation:
    def __init__(self, mean_values, std_values, log_values):

        self.mean = mean_values
        self.std = std_values
        self.log = log_values
    
    def transform(self, x):

        y = np.zeros(x.shape)
        
        for i in self.log:
            y[..., i] = np.log(x[...,i])

        return (y - self.mean)/self.std
        
    def untransform(self, x):
        
        y = self.mean + self.std*x
        
        for i in self.log:
            y[..., i] = np.exp(y[..., i])
        
        return np.maximum(y, 0.0)
