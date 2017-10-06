from .BaseNode import *

import tensorflow as tf
import numpy as np

class OutputNodeRegression(BaseNode):
    def __init__(self, target_placeholder, input_means, input_vars):
        BaseNode.__init__(self, input_means, input_vars)
        self.target_placeholder = target_placeholder
        self.input_means = input_means
        self.input_vars = input_vars
        self.output_means = input_means
        self.output_vars = input_vars

    def getEnergyContribution(self):
        z = -0.5 * tf.log(2.0 * np.pi * self.input_vars)
        exp = - 0.5 * tf.square(self.target_placeholder - self.input_means) \
            / self.input_vars
        return  tf.reduce_sum(tf.reduce_sum(z + exp, 1, keep_dims=True),
                               0,
                               keep_dims=True)

    def getOutput(self):
        return self.output_means, self.output_vars


