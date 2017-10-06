from ..AbstractModel import AbstractModel
import GPy

import time
import tensorflow as tf
import numpy as np

class GPlib(AbstractModel):
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
        
        self.kern = 'rbf'
        if ('kern' in modelParams):
            self.kern = modelParams['kern']
        
        self.reset()
        
    def reset(self):
        self.models = []
        for i in range(0, self.output_d):
            if (self.kern == 'matern'):
                kernel = GPy.kern.Matern52(input_dim=self.input_d, ARD=True)  
            else:
                kernel = GPy.kern.RBF(input_dim=self.input_d, ARD=True)  
            model = GPy.models.GPRegression(self.training_data, self.training_targets[:, i:i+1],kernel)
            model.optimize_restarts(num_restarts = 10)
            model.optimize(messages=False)
            #print(kernel)
            self.models.append(model)
        

    def addPoint(self, x, y):
        self.training_data = np.vstack((x, self.training_data))
        self.training_targets = np.vstack((y, self.training_targets))
        self.reset()

    def predictBatch(self, test_data):
        means = np.array([[]]*test_data.shape[0])
        vars = np.array([[]]*test_data.shape[0])
        for model in self.models:
            mean, var = model.predict(test_data, full_cov=False)
            means = np.concatenate((means, mean), axis=1)
            vars = np.concatenate((vars, var.reshape((-1, 1))), axis=1)
        return means, vars
