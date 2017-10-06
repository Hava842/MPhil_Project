from ..AbstractModel import AbstractModel
from .GPNetwork import GPNetwork

import time
import tensorflow as tf
import numpy as np

class GPNetworkDJ(AbstractModel):
    # The training_targets are the Y's which are real numbers
    def __init__(self, 
                 training_data,
                 training_targets,
                 modelParams):
        
        self.training_data = training_data
        self.training_targets = training_targets
        self.n_points = training_data.shape[0]
        self.input_d = training_data.shape[1]
        self.output_d = training_targets.shape[1]
        
        self.models = []
        for i in range(0, self.output_d):
            model = GPNetwork(self.training_data, self.training_targets[:, i:(i+1)], modelParams)
            self.models.append(model)
        

    def addPoint(self, x, y):
        for i, model in enumerate(self.models):
            model.addPoint(x, y[:, i:(i+1)])

    def predictBatch(self, test_data):
        means = np.array([[]]*test_data.shape[0])
        vars = np.array([[]]*test_data.shape[0])
        for model in self.models:
            mean, var = model.predictBatch(test_data)
            means = np.concatenate((means, mean), axis=1)
            vars = np.concatenate((vars, var), axis=1)
        return means, vars
