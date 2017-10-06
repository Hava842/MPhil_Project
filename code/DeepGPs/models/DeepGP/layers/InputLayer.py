from .BaseLayer import BaseLayer
from ..nodes.InputNode import *

import tensorflow as tf
import numpy as np

class InputLayer(BaseLayer):
    def __init__(self, data_placeholder):
        self.input_means = data_placeholder
        self.input_vars = tf.zeros_like(data_placeholder)

        BaseLayer.__init__(self, self.input_means, self.input_vars)
        
        self.input_node = InputNode(self.input_means, self.input_vars)
        self.output_means, self.output_vars = self.input_node.getOutput()

    def getEnergyContribution(self):
        return self.input_node.getEnergyContribution()

    def getOutput(self):
        return self.output_means, self.output_vars


