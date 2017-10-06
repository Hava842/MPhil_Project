
import numpy as np
import argparse
from sklearn import preprocessing
from scipy.stats import multivariate_normal
import math
import matplotlib
matplotlib.use('Agg')

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
    
    
    
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('model', help='shallow/deep_joint/deep_disjoint')
parser.add_argument('dir', help='shallow/deep_joint/deep_disjoint')
parser.add_argument('-rs', dest='seed', type=int, default=422,
                    help='random seed')
                    
args = parser.parse_args()

import random
random.seed(args.seed)
np.random.seed(args.seed)

n_prints = 1
deep_maxiter = 15000

if (args.model == 'shallow'):
    print('Shallow GP')
    modelParams = {'model':'dgp',
                   'maxiter': 2000,
                   'layer_types': ['gp', 'noisy'],
                   'layer_nodes': [2, 1]}
elif (args.model == 'shallow_disjoint'):
    print('Shallow disjoint GP')
    modelParams = {'model':'dgp_dj',
                   'maxiter': 2000,
                   'layer_types': ['gp', 'noisy'],
                   'layer_nodes': [2, 1]}
elif (args.model == 'deep_joint'):
    print('Deep joint GP')
    modelParams = {'model':'dgp',
                   'maxiter': 2000,
                   'layer_types': ['gp', 'noisy', 'gp', 'noisy'],
                   'layer_nodes': [2, 1, 2, 1]}
elif (args.model == 'deep_disjoint'):
    print('Deep disjoint GP')
    modelParams = {'model':'dgp_dj',
                   'maxiter': 2000,
                   'layer_types': ['gp', 'noisy', 'gp', 'noisy'],
                   'layer_nodes': [2, 1, 2, 1]}
elif (args.model == 'dgps'):
    print('dgps model')
    modelParams = {'model':'dgps'}
elif (args.model == 'dgps_disjoint'):
    print('dj dgps model')
    modelParams = {'model':'dgps_dj'}
elif (args.model == 'dgps_shallow'):
    print('shallow dgps model')
    modelParams = {'model':'dgps',
                   'shallow': True}
elif (args.model == 'gp'):
    print('GP')
    modelParams = {'model':'gp',
                   'maxiter': 40000,
                   'retrain': args.retrain}
elif (args.model == 'gplib'):
    print('GPlib')
    modelParams = {'model':'gplib'}
elif (args.model == 'gplib_matern'):
    print('GPlib matern')
    modelParams = {'model':'gplib',
                   'kern': 'matern'}
elif (args.model == 'gp_disjoint'):
    print('Disjoint GP')
    modelParams = {'model':'gp_dj',
                   'maxiter': 5000}
elif (args.model == 'random'):
    print('Random')
    modelParams = {'model':'rnd'}
       
filename = '{}/{}_{}.txt'.format(args.dir, args.model, args.seed)
with open(filename, 'w+') as file:
    pass
    
filename_rmse = '{}/{}_rmse_{}.txt'.format(args.dir, args.model, args.seed)
with open(filename_rmse, 'w+') as file:
    pass
       
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
       
for n_points in [0]:
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
    if (modelParams['model'] == 'dgp'):
        model = GPNetwork(init_points, init_values, modelParams)
    elif (modelParams['model'] == 'dgp_dj'):
        model = GPNetworkDJ(init_points, init_values, modelParams)
    elif (modelParams['model'] == 'gp'):
        model = GaussianProcess(init_points, init_values, modelParams)
    elif (modelParams['model'] == 'gp_dj'):
        model = GaussianProcessDJ(init_points, init_values, modelParams)
    elif (modelParams['model'] == 'rnd'):
        model = RandomModel(init_points, init_values, modelParams)
    elif (modelParams['model'] == 'dgps'):
        model = Dgps_net(init_points, init_values, modelParams)
    elif (modelParams['model'] == 'dgps_dj'):
        model = Dgps_netDJ(init_points, init_values, modelParams)
    elif (modelParams['model'] == 'gplib'):
        model = GPlibDJ(init_points, init_values, modelParams)
    else:
        print('Unspecified model name')
              

    #for x in range(0, n_prints):
    means, vars = model.predictBatch(frontier_points)
    
    llsum = 0.0
    llsep = [0.0, 0.0]
    rmsesum =  0.0
    rmsesep = [0.0, 0.0]
    for i in range(0, frontier_points.shape[0]):
        for j in range(0, 2):
            llsum += multivariate_normal.logpdf(frontier[i, j], mean=means[i, j], cov=vars[i, j])
            rmsesum += (means[i, j] - frontier[i, j])**2
            llsep[j] += multivariate_normal.logpdf(frontier[i, j], mean=means[i, j], cov=vars[i, j])
            rmsesep[j] += (means[i, j] - frontier[i, j])**2
    #           file.write("{0} mean {1}, var {2}, actual {3}\n".format(labels[j], means[i, j], vars[i, j], frontier[i, j]))
    print("Accuracy {0}, Power {1}".format(llsep[0] / frontier_points.shape[0], llsep[1] / frontier_points.shape[0]))
    print("After {0} points, the avg log likelihood is {1}".format(n_points, 0.5 * llsum / frontier_points.shape[0]))
    #    model.train()
        
    with open(filename, 'a') as file:
        file.write("{0} {1} {2} {3} \n".format(n_points, 0.5 * llsum / frontier_points.shape[0], llsep[0] / frontier_points.shape[0], llsep[1] / frontier_points.shape[0]))
    
    with open(filename_rmse, 'a') as file:
        file.write("{0} {1} {2} {3} \n".format(n_points, np.sqrt(rmsesum * 0.5 / frontier_points.shape[0]), np.sqrt(rmsesep[0] / frontier_points.shape[0]), np.sqrt(rmsesep[1] / frontier_points.shape[0])))
