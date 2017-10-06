from tabulate import tabulate
import numpy as np
import re

names = ['dgps', 'dgps_shallow', 'dgps_disjoint', 'gplib', 'gplib_matern']



for ext in ['', '_rmse']:
    table = [['model'] + [20, 50, 100, 200, 300] * 3]
    for name in names:
        ll = {'20':[],
              '50':[],
              '100':[],
              '200':[],
              '300':[]}
        acc = {'20':[],
              '50':[],
              '100':[],
              '200':[],
              '300':[]}
        eng = {'20':[],
              '50':[],
              '100':[],
              '200':[],
              '300':[]}
        for i in range(1, 100):
            with open('./predictor_rmse/{}{}_{}.txt'.format(name, ext, str(i)), 'r') as f:
                content = f.readlines()
            
            for line in content:
                lik = []
                split = line.split(' ')
                for term in split:
                    try:
                        lik.append(float(term))
                    except:
                        pass
                
                if (lik[1] < 40.0 and lik[1] > -40.0):
                    ll[split[0]].append(lik[1])
                    acc[split[0]].append(lik[2])
                    eng[split[0]].append(lik[3])
                
        line = [name]
        for list in [acc, eng, ll]:
            for x in ['20', '50', '100', '200', '300']:
                #print(len(list[x]))
                mean = 0.0
                for l in list[x]:
                    mean += l / len(list[x])
                sigmas = 0.0
                for l in list[x]:
                    sigmas += (mean - l)**2 / (len(list[x]) - 1.0)
                line.append('{0:.2f} +- {1:.2f}'.format(mean, 1.984 * np.sqrt(sigmas / len(list[x]))))
        table.append(line)
    table = np.array(table)
    table = np.transpose(table)
    print(table)
    print(tabulate(table, tablefmt="latex"))
        