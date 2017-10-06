import abc

class BaseLayer(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def getEnergyContribution(self):
        return 0.0

    @abc.abstractmethod
    def getOutput(self):
        return 0.0, 0.0

    def __init__(self, input_means, input_vars):
        self.input_means = input_means
        self.input_vars = input_vars

