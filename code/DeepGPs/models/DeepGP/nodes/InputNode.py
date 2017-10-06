from .BaseNode import *

import tensorflow as tf
import numpy as np

class InputNode(BaseNode):
    def __init__(self, input_means, input_vars):
        BaseNode.__init__(self, input_means, input_vars)
        self.data_placeholder = input_means

    def getOutput(self):
        return self.data_placeholder, tf.zeros_like(self.data_placeholder)

    def getEnergyContribution(self):
        return tf.constant(0.0, tf.float32, [1, 1])