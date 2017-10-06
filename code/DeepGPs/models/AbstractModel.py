import abc

class AbstractModel(object):
  
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(X, Y, modelParams):
        pass

    @abc.abstractmethod
    def addPoint(x, y):
        pass

    @abc.abstractmethod
    def predictBatch(X):
        pass


