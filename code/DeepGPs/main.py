import matplotlib
matplotlib.use('Agg')
from optimizer.Optimizer import optimize
from dgplot import prettyplot

import numpy as np
# import matplotlib.pyplot as plt
import sys

def fcliff(x):
    result = []
    tmp = 0.9*np.exp(-3.0*np.dot(x, x))
    if (x[0] <= 0.0):
        return np.array([[-tmp]])
    else:
        return np.array([[tmp]])

# prettyplot(fcliff)

def f(x):
    return np.array([(x-0.5)**2, (x+0.5)**2])

from sklearn import preprocessing
from scipy.stats import multivariate_normal

from models.DeepGP.GPNetwork import GPNetwork
from models.DeepGP.GPNetworkDJ import GPNetworkDJ
from models.GP.GaussianProcess import GaussianProcess
from models.GP.GaussianProcessDJ import GaussianProcessDJ
from models.Random.RandomModel import RandomModel
from models.dgps.dgps_net import Dgps_net
from models.dgps.dgps_net_dj import Dgps_netDJ
from models.GP.GPlibDJ import GPlibDJ


with open("/home/hava/MPhil_Project/code/DeepGPs/random_evaluations.txt", "r") as re:
    re_proc = []
    iss = []
    oss = []
    for line in re:
        line_split = line.split(' ')
        i = [float(x) for x in line_split[0:13]]
        o = [float(x) for x in line_split[13:15]]
        iss.append(i)
        oss.append(o)
    iss = np.array(iss)
    oss = np.array(oss)
    iss = preprocessing.scale(iss)
    #oss = preprocessing.scale(oss)
    for i in range(0, iss.shape[0]):
        re_proc.append((iss[i, :], oss[i, :]))
    re_proc = np.array(re_proc)
    print(re_proc)

def fre(x):
    for i, o in re_proc:
        if (np.allclose(i, x, atol=1e-08)):
            return o
    assert(True)
    
def freinv(x):
    for i, o in re_proc:
        if (np.allclose(o, x, atol=1e-08)):
            return i
    assert(True)

modelParams = {'model':'gplib'}
                   
aquisitionParams = {'aquisition':'SMSego',
                    'gain': 2.0}

frontier, curve = optimize(f, np.array([[i] for i in np.linspace(-1, 1, 500)]), modelParams, aquisitionParams, 5, 1)

# plt.figure(3)
# ax = plt.gca()
# plt.gca().grid(True)
# ind = frontier[:, 0].argsort()
# frontier = frontier[ind, :]
# plt.plot(frontier[:, 0], frontier[:, 1], 'gs')
# flist = []
# current_point = [0, 10]
# for i in range(0, len(frontier)):
    # current_point[0] = frontier[i, 0]
    # flist.append(current_point)
    # flist.append(frontier[i, :])
    # current_point = current_point.copy()
    # current_point[1] = frontier[i, 1]
# current_point[0] = 10
# flist.append(current_point)
# flist = np.array(flist)
# plt.plot(flist[:, 0], flist[:, 1], 'g-')    
# ax.fill_between(flist[:, 0], 10, flist[:, 1], facecolor='green', alpha=0.5, hatch='//')
# plt.xlabel('Obj 1')
# plt.ylabel('Obj 2')
# plt.title('Pareto frontier')
# plt.draw()

# plt.figure()
# plt.gca().grid(True)
# plt.plot(range(0, len(curve)), curve, 'bo-')
# plt.ylabel('Hypervolume')
# plt.xlabel('Iterations')
# plt.title('Increasing hypervolume over the iterations')
# plt.savefig('curve.eps')

# used = np.array(used)
# plt.figure()
# plt.gca().grid(True)
# plt.scatter(frontier[:, 0], frontier[:, 1], c='k', marker='s')
# plt.scatter(used[:, 0], used[:, 1], c='k', marker='x')
# plt.xlabel('Accuracy')
# plt.ylabel('Power consumption')
# plt.title('Evaluated points')

# plt.savefig('points.eps')
# plt.show()


