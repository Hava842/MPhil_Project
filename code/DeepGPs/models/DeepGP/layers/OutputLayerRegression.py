from .BaseLayer import *
from ..nodes.OutputNodeRegression import *

import tensorflow as tf
import numpy as np

class OutputLayerRegression(BaseLayer):
    def __init__(self, target_placeholder, input_means, input_vars):
        BaseLayer.__init__(self, input_means, input_vars)
        
        self.output_node = OutputNodeRegression(target_placeholder,
                                                input_means,
                                                input_vars)
        self.output_means, self.output_vars = self.output_node.getOutput()

    def getEnergyContribution(self):
        return self.output_node.getEnergyContribution()

    def getOutput(self):
        return self.output_means, self.output_vars


