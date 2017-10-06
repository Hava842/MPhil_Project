import abc

class AbstractAquisition(object):
  
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def getAquisitionBatch(X, model, existingY):
        pass

    