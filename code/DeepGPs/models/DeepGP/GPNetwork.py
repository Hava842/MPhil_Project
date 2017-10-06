from .layers.InputLayer import *
from .layers.OutputLayerRegression import *
from .layers.OutputLayerRegressionMultioutput import *
from .layers.NoisyLayer import *
from .layers.GPLayer import *
from ..AbstractModel import AbstractModel

import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class GPNetwork(AbstractModel):
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

        self.maxiter = 1500
        if ('maxiter' in modelParams):
            self.maxiter = modelParams['maxiter']

        self.layer_types = ['gp', 'gp']
        if ('layer_types' in modelParams):
            self.layer_types = modelParams['layer_types']

        self.layer_nodes = [self.input_d, self.output_d]
        if ('layer_nodes' in modelParams):
            self.layer_nodes = modelParams['layer_nodes']
        
        self.minibatch_size = 500
        if ('minibatch_size' in modelParams):
            self.minibatch_size = modelParams['minibatch_size']

        self.learning_rate = 0.01
        if ('learning_rate' in modelParams):
            self.learning_rate = modelParams['learning_rate']
            
        self.retrain = 0
        if ('retrain' in modelParams):
            self.retrain = modelParams['retrain']
        self.retrain_counter = 0
        
        self.early_stopping = False
        if ('early_stopping' in modelParams):
            self.early_stopping = modelParams['early_stopping']
        
        self.decay_lr = False
        if ('decay_lr' in modelParams):
            self.decay_lr = modelParams['decay_lr']
            
        self.resetGraph()

        print("Start training")
        self.train()

    def addPoint(self, x, y):
        self.training_data = np.vstack((x, self.training_data))
        self.training_targets = np.vstack((y, self.training_targets))
        self.n_points += 1
        self.session.run(self.n_points_tf.assign(self.n_points))
        self.train()

    def predictBatch(self, test_data):
        self.session.run(self.set_for_training.assign(0.0))
        fd = {self.data_placeholder:test_data}
        return self.session.run((self.output_mean, self.output_var),
                                feed_dict=fd)

    def addInputLayer(self):
        assert len(self.layers) == 0

        self.layers.append(InputLayer(self.data_placeholder))

    def addNoisyLayer(self):
        assert len(self.layers) != 0

        means, vars = self.layers[-1].getOutput()
        new_layer = NoisyLayer(means, vars)
        self.layers.append(new_layer)

    def addGPLayer(self, n_inducing_points, n_nodes=1, initial=None):
        assert len(self.layers) != 0

        means, vars = self.layers[-1].getOutput()
        new_layer = GPLayer(self.n_points_tf,
                            n_inducing_points,
                            n_nodes,  
                            means,
                            vars,
                            self.set_for_training,
                            initial)
        self.layers.append(new_layer)

    def addOutputLayerRegression(self):
        assert len(self.layers) != 0

        means, vars = self.layers[-1].getOutput()
        new_layer = OutputLayerRegression(self.target_placeholder,
                                          means,
                                          vars)
        self.layers.append(new_layer)
        self.output_mean, self.output_var = new_layer.getOutput()

    def resetGraph(self):
        tf.reset_default_graph()
        
        print('Initializing computation graphs')
        
        self.n_points_tf = tf.Variable(self.n_points, 
                                       trainable=False,
                                       dtype=tf.float32)
        self.set_for_training = tf.Variable(1.0,
                                            trainable=False,
                                            dtype=tf.float32)
        self.data_placeholder = tf.placeholder(tf.float32,
                                               [None, self.input_d])
        self.target_placeholder = tf.placeholder(tf.float32,
                                                 [None, self.output_d])
                        
        self.layers = []
        
        self.addInputLayer()
        for l in range(0, len(self.layer_types)):
            print('Layer {0}'.format(l))
            gp_points = max(int(np.ceil(0.1 * self.n_points)), 5)
            if (l == 0):
                self.addGPLayer(gp_points,
                                self.layer_nodes[l],
                                initial=self.training_data)
            elif (self.layer_types[l] == 'gp'):
                self.addGPLayer(gp_points, self.layer_nodes[l])
            elif (self.layer_types[l] == 'noisy'):
                self.addNoisyLayer()
        self.addOutputLayerRegression()

        layer_energies = [l.getEnergyContribution() for l in self.layers]
        self.energy = tf.add_n(layer_energies)

        if (self.decay_lr):
            global_step = tf.Variable(0, trainable=False)
            boundaries = [1500, 3000, 6000]
            values = [0.01, 0.003, 0.001, 0.0003]
            self.actual_learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
            adam = tf.train.AdamOptimizer(self.actual_learning_rate)
            self.optimizer = adam.minimize(-self.energy, global_step=global_step)
        else:
            adam = tf.train.AdamOptimizer(self.learning_rate)
            self.optimizer = adam.minimize(-self.energy)
        
        print("Initializing variables")
        init_op = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(init_op)

    def train(self):
        self.retrain_counter += 1
        print('{0} iterations until retrain'.format(self.retrain - self.retrain_counter))
        if (self.retrain_counter == self.retrain):
            self.retrain_counter = 0
            self.resetGraph()
        
        if (self.decay_lr):
            print('Learning rate: %f' % (self.session.run(self.actual_learning_rate)))
            
        self.session.run(self.set_for_training.assign(1.0))

        n_batches = int(np.ceil(1.0 * self.n_points / self.minibatch_size))
        last_energy = 0.0
        failed_to_improve = False
        for iter in range(self.maxiter):
            suffle = np.random.permutation(self.n_points)
            training_data = self.training_data[ suffle, : ]
            training_targets = self.training_targets[ suffle, : ]
            start_epoch  = time.time()
            epoch_energy = 0.0
            for i in range(n_batches):
                start_i = i * self.minibatch_size
                end_i = min((i + 1) * self.minibatch_size, self.n_points)
                minibatch_data = training_data[start_i : end_i, : ]
                minibatch_targets = training_targets[start_i : end_i, : ]
                fd = {
                      self.data_placeholder:minibatch_data,
                      self.target_placeholder:minibatch_targets
                     }

                _, e = self.session.run((self.optimizer, self.energy),
                                        feed_dict=fd)
                epoch_energy += e

            if (iter % 50 == 0):
                print('Epoch: {}, - Energy: {} Time: {}'
                        .format(iter, epoch_energy, time.time() - start_epoch))
                if (last_energy >= epoch_energy and failed_to_improve and self.early_stopping):
                    print('Early stopping')
                    break
                else:
                    failed_to_improve = (last_energy >= epoch_energy)
                    last_energy = epoch_energy
