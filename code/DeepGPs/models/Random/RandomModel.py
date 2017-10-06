from ..AbstractModel import AbstractModel

import numpy as np

class RandomModel(AbstractModel):
    def __init__(self, 
                 training_data,
                 training_targets,
                 modelParams):
        self.dim = training_targets.shape[1]
        
    def addPoint(self, x, y):        
        pass
        
    def predictBatch(self, X_): 
        return np.random.rand(X_.shape[0], self.dim), np.random.rand(X_.shape[0], self.dim)
