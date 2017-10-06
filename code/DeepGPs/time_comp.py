
import numpy as np
import argparse
from sklearn import preprocessing
from scipy.stats import multivariate_normal
import time
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    
modelParams2 = {'model':'dgps'}
modelParams1 = {'model':'dgp',
               'maxiter': 1,
               'layer_types': ['gp', 'noisy', 'gp', 'noisy'],
               'layer_nodes': [2, 1, 2, 1],
               'early_stopping': False}
       
doms = []
for i in range(0, iss.shape[0]):
    current = 0
    for j in range(0, iss.shape[0]):
        if (oss[i, 0] > oss[j, 0] and oss[i, 1] > oss[j, 1]):
            current += 1
    doms.append(current)
doms = np.array(doms)
dom_inds = doms.argsort()
print(doms[dom_inds])
       
for n_points in [300]:
    #with open('random_points.txt') as file:
    #    init_points = []
    #    for line in file: # read rest of lines
    #        init_points.append([float(x) for x in line.split()])
    #    init_points = np.array(init_points)
    train_ind = np.random.choice(dom_inds[0:1200], size=n_points)
    init_points = iss[train_ind, :]
    
    #with open('random_frontier.txt') as file:
    #    frontier = []
    #    for line in file: # read rest of lines
    #        line = line[1:-2]
    #        frontier.append([float(x) for x in line.split()])
    #    frontier_points = np.array([freinv(np.array(l)) for l in frontier])
    test_ind = [x for x in dom_inds[0:1200] if x not in train_ind]
    frontier_points = iss[test_ind, :]
    frontier = oss[test_ind, :]

    init_values = []
    for point in init_points:
        init_values.append(fre(point))
    init_values = np.reshape(np.array(init_values), (len(init_values), -1))
        
    # Model
    start = time.time()
    model = GPNetwork(init_points, init_values, modelParams1)
    tftime = time.time() - start
    
    #start = time.time()
    #model2 = Dgps_net(init_points, init_values, modelParams2)
    #theanotime = time.time() - start
              
    ll_train = []
    ll_test = []
    energy = []
    for i in range(0, 2000):
        means, vars = model.predictBatch(frontier_points)
        #if (np.isnan(vars).any() or (vars <= 0.0).any()):
        #    for asd in range(0, 12):
        #        print('PROBLEM')
        #        model.train()
        llsum = 0.0
        llsep = [0.0, 0.0]
        rmsesum =  0.0
        rmsesep = [0.0, 0.0]
        for i in range(0, frontier_points.shape[0]):
            for j in range(0, 2):
                llsum += multivariate_normal.logpdf(frontier[i, j], mean=means[i, j], cov=vars[i, j])
                rmsesum += (means[i, j] - frontier[i, j])**2
                #llsep[j] += multivariate_normal.logpdf(frontier[i, j], mean=means[i, j], cov=vars[i, j])
                #rmsesep[j] += (means[i, j] - frontier[i, j])**2
        #           file.write("{0} mean {1}, var {2}, actual {3}\n".format(labels[j], means[i, j], vars[i, j], frontier[i, j]))
        print("After {0} points, the avg log likelihood is {1}".format(n_points, 0.5 * llsum / frontier_points.shape[0]))
        ll_test.append(0.5 * llsum / frontier_points.shape[0])
        
        means, vars = model.predictBatch(init_points)
        llsum = 0.0
        llsep = [0.0, 0.0]
        rmsesum =  0.0
        rmsesep = [0.0, 0.0]
        for i in range(0, init_points.shape[0]):
            for j in range(0, 2):
                llsum += multivariate_normal.logpdf(init_values[i, j], mean=means[i, j], cov=vars[i, j])
                rmsesum += (means[i, j] - init_values[i, j])**2
                llsep[j] += multivariate_normal.logpdf(init_values[i, j], mean=means[i, j], cov=vars[i, j])
                rmsesep[j] += (means[i, j] - init_values[i, j])**2
        #           file.write("{0} mean {1}, var {2}, actual {3}\n".format(labels[j], means[i, j], vars[i, j], frontier[i, j]))
        ll_train.append(0.5 * llsum / init_points.shape[0])
        print("After {0} points, the avg log likelihood is {1}".format(n_points, 0.5 * llsum / init_points.shape[0]))
        
        energy.append(model.get_energy())
        model.train()
    
    plt.figure()
    plt.plot(np.linspace(1, 2000, num=2000), ll_train, label='Training log-likelihood')
    plt.plot(np.linspace(1, 2000, num=2000), ll_test, label='Test log-likelihood')
    plt.ylabel('Log-likelihood')
    plt.ylim(-4, 0)
    plt.xlabel('Iterations')
    plt.title('Training and Test log-likelihoods')
    plt.legend()
    
    serial = np.random.randint(100000)
    plt.savefig('figs/ll_{}_{}.eps'.format(n_points, serial))
    plt.savefig('figs/ll_{}_{}.pdf'.format(n_points, serial))
    
    plt.figure()
    plt.plot(np.linspace(1, 2000, num=2000), np.reshape(energy, (-1)))
    plt.ylabel('Energy')
    plt.ylim(-1500, 500)
    plt.xlabel('Iterations')
    plt.title('Model energy')
    plt.savefig('figs/en_{}_{}.eps'.format(n_points, serial))
    plt.savefig('figs/en_{}_{}.pdf'.format(n_points, serial))
    #with open("times.txt", "a") as myfile:
    #    myfile.write("{} tf time: {}, thenao time: {}".format(n_points, tftime, theanotime))