from .BaseLayer import *
from ..nodes.NoisyNode import *

import tensorflow as tf
import numpy as np

class NoisyLayer(BaseLayer):
    def __init__(self, input_means, input_vars):

        BaseLayer.__init__(self, input_means, input_vars)
        
        self.noisy_node = NoisyNode(input_means, input_vars)
        self.output_means, self.output_vars = self.noisy_node.getOutput()

    def getEnergyContribution(self):
        return self.noisy_node.getEnergyContribution()

    def getOutput(self):
        return self.output_means, self.output_vars
