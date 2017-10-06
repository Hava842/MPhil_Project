from .BaseLayer import *
from ..nodes.GPNode import *

import tensorflow as tf
import numpy as np

class GPLayer(BaseLayer):
    def __init__(self,
                 n_points,
                 n_inducing_points,
                 n_nodes, input_means,
                 input_vars,
                 set_for_training,
                 initial=None):
        BaseLayer.__init__(self, input_means, input_vars)
        self.nodes = []
        self.output_means_list = []
        self.output_vars_list = []

        for i in range(n_nodes):
            if (initial is None):
                initial_sample = None
            else:
                indices = np.random.choice(initial.shape[0], size=n_inducing_points)
                initial_sample = initial[indices, :]
            gp_node = GPNode(input_means,
                             input_vars,
                             n_points,
                             n_inducing_points,
                             set_for_training,
                             initial=initial_sample)
            output_mean, output_var = gp_node.getOutput()
            self.output_means_list.append(output_mean)
            self.output_vars_list.append(output_var)
            self.nodes.append(gp_node)
        
        self.output_means = tf.concat(self.output_means_list, 1)
        self.output_vars = tf.concat(self.output_vars_list, 1)

        self.energy = tf.add_n([n.getEnergyContribution() for n in self.nodes])

    def getEnergyContribution(self):
        return self.energy

    def getOutput(self):
        return self.output_means, self.output_vars


