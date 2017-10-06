from .BaseNode import *

import tensorflow as tf
import numpy as np

class NoisyNode(BaseNode):
    def __init__(self, input_means, input_vars):
        BaseNode.__init__(self, input_means, input_vars)
        self.input_means = input_means
        self.input_vars = input_vars
        input_d = self.input_means.get_shape().as_list()[1]

        self.log_noise = tf.Variable(tf.zeros([1, input_d]))

        self.output_means = self.input_means
        self.output_vars = input_vars + tf.exp(self.log_noise)

    def getEnergyContribution(self):
        return tf.constant(0.0, tf.float32, [1, 1])

    def getOutput(self):
        return self.output_means, self.output_vars