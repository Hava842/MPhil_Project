import matplotlib
matplotlib.use('Agg')

from optimizer.Optimizer import optimize

import numpy as np
import argparse
from sklearn import preprocessing

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
    #iss = preprocessing.scale(iss)
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
    
    
    
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('model', help='shallow/deep_joint/deep_disjoint')
parser.add_argument('output_dir')
parser.add_argument('iterations', type=int)
parser.add_argument('-rs', dest='seed', type=int, default=np.random.random_integers(5000),
                    help='random seed')
parser.add_argument('-rt', dest='retrain', type=int, default=1,
                    help='retrain frequency')
                    
args = parser.parse_args()

import random
random.seed(args.seed)
np.random.seed(args.seed)

if (args.model == 'shallow'):
    print('Shallow GP')
    modelParams = {'model':'dgp',
                   'maxiter': 15000,
                   'layer_types': ['gp'],
                   'layer_nodes': [2],
                   'retrain': args.retrain}
elif (args.model == 'shallow_disjoint'):
    print('Shallow disjoint GP')
    modelParams = {'model':'dgp_dj',
                   'maxiter': 15000,
                   'layer_types': ['gp'],
                   'layer_nodes': [1],
                   'retrain': args.retrain}
elif (args.model == 'deep_joint'):
    print('Deep joint GP')
    modelParams = {'model':'dgp',
                   'maxiter': 15000,
                   'layer_types': ['gp', 'gp'],
                   'layer_nodes': [2, 2],
                   'retrain': args.retrain}
elif (args.model == 'deep_disjoint'):
    print('Deep disjoint GP')
    modelParams = {'model':'dgp_dj',
                   'maxiter': 15000,
                   'layer_types': ['gp', 'gp'],
                   'layer_nodes': [2, 1],
                   'retrain': args.retrain}
elif (args.model == 'dgps'):
    print('dgps model')
    modelParams = {'model':'dgps'}
elif (args.model == 'gp'):
    print('GP')
    modelParams = {'model':'gp',
                   'maxiter': 40000,
                   'retrain': args.retrain}
elif (args.model == 'gp_disjoint'):
    print('Disjoint GP')
    modelParams = {'model':'gp_dj',
                   'maxiter': 40000,
                   'retrain': args.retrain}
elif (args.model == 'random'):
    print('Random')
    modelParams = {'model':'rnd'}
                   
aquisitionParams = {'aquisition':'SMSego',
                    'gain': 2.0}
                    

frontier, curve = optimize(fre, np.array([i for i, o in re_proc]), modelParams, aquisitionParams, 50, args.iterations, args.output_dir)




