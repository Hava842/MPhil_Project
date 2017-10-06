from .AbstractAquisition import AbstractAquisition

import numpy as np

class SMSego(AbstractAquisition):
    def __init__(self, aquisitionParams):
        self.gain = 1.0
        if ('gain' in aquisitionParams):
            self.gain = aquisitionParams['gain']

        self.epsilon = 1e-6
        if ('epsilon' in aquisitionParams):
            self.epsilon = aquisitionParams['epsilon']

        self.n_dim = 2
        if ('n_dim' in aquisitionParams):
            self.n_dim = aquisitionParams['n_dim']
        
        self.reference = np.array([20, 40])
        if ('reference' in aquisitionParams):
            self.reference = aquisitionParams['reference']
    
    def getGoalName(self):
        return 'Hypervolume'
    
    def getGoalValue(self, frontier):
        return self.getVolume(frontier)

    # This function is exponential in the number of dimensions
    def getVolume(self, Y):
        return self.getVolumeRecursive(Y, 0)

    def getVolumeRecursive(self, Y, dim):
        if (dim == Y.shape[1]-1):
            return self.reference[dim] - min(Y[:, dim])
        
        sortedY = np.array(sorted(Y, key=lambda Y_entry: -Y_entry[dim]))
        
        accumulator = 0.0
        sweep = self.reference[dim]
        while (sortedY.shape[0] > 0):
            accumulator += (sweep - sortedY[0, dim]) \
                           * self.getVolumeRecursive(sortedY, dim+1)
            sweep = sortedY[0, dim]
            sortedY = sortedY[1:, :]
        
        return accumulator

    def getAquisitionBatch(self, X, model, frontier):
        n_points = X.shape[0]

        means, vars = model.predictBatch(X)
        pot_sol = means - self.gain * np.sqrt(vars)
        
        hv_frontier = self.getVolume(frontier)
        aquisitions = np.ones((n_points))
        for i in range(0, n_points):
            penalty = 0.0
            for k in range(0, frontier.shape[0]):
                if np.all(frontier[k, :] <= pot_sol[i, :] + self.epsilon):
                    p = -1 + np.prod(1 +
                                     np.maximum(pot_sol[i, :] - frontier[k, :],
                                                np.zeros_like(pot_sol[i, :]))
                                    )
                    penalty = np.maximum(penalty, p)

            if (penalty == 0.0):
                hv_pot = self.getVolume(np.vstack((pot_sol[i, :], frontier)))
                aquisitions[i] = -hv_frontier + hv_pot
            else:
                aquisitions[i] = -penalty
        
        return aquisitions


