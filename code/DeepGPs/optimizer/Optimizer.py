from models.DeepGP.GPNetwork import GPNetwork
from models.DeepGP.GPNetworkDJ import GPNetworkDJ
from models.GP.GPlib import GPlib
from models.Random.RandomModel import RandomModel
from .aquisition.SMSego import SMSego

import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import os.path

def optimize(f,
             candidates,
             modelParams,
             aquisitionParams,
             init_eval,
             max_eval,
             output_dir,
             plots=False):
    
    print('Initializing models')   

    # Points of the initial evaluations
    if (os.path.isfile('{0}/points.txt'.format(output_dir))):
        with open('{0}/points.txt'.format(output_dir)) as file:
            init_points = []
            for line in file: # read rest of lines
                init_points.append([float(x) for x in line.split()])
        init_points = np.array(init_points)
        
        with open('{0}/candidates.txt'.format(output_dir)) as file:
            candidates = []
            for line in file: # read rest of lines
                candidates.append([float(x) for x in line.split()])
        candidates = np.array(candidates)
    else:
        init_index = np.random.randint(0, candidates.shape[0], (init_eval))
        init_points = candidates[init_index, :]
        candidates = np.delete(candidates, init_index, 0)
        
    print('Initital Points')
    init_values = []
    for point in init_points:
        init_values.append(f(point))
    init_values = np.reshape(np.array(init_values), (len(init_values), -1))
    print(init_values)


    # Model
    if (modelParams['model'] == 'dgp'):
        model = GPNetwork(init_points, init_values, modelParams)
    elif (modelParams['model'] == 'dgp_dj'):
        model = GPNetworkDJ(init_points, init_values, modelParams)
    elif (modelParams['model'] == 'gp'):
        model = GPlib(init_points, init_values, modelParams)
    elif (modelParams['model'] == 'rnd'):
        model = RandomModel(init_points, init_values, modelParams)
    else:
        print('Unspecified model name')

    # Aquisition
    if (aquisitionParams['aquisition'] == 'SMSego'):
        aquisition_function = SMSego(aquisitionParams)
    else:
        print('Unspecified aquisition function')

    # Iteration
    frontier = find_frontier(init_values)    
    if (os.path.isfile('{0}/curve.txt'.format(output_dir))):
        curve = []
        with open('{0}/curve.txt'.format(output_dir)) as file:
            for line in file: # read rest of lines
                curve.append(float(line))
    else:
        curve = [aquisition_function.getGoalValue(frontier)]

    for iter in range(0, max_eval):
        iter_start = time.time()
    
        print('Predicting')
        pred_means, pred_vars = model.predictBatch(candidates)
        #print(pred_vars)
        pred_vars = np.sqrt(pred_vars)
        
        if (plots):
            plt.figure(1)
            plt.clf()
            plt.gca().grid(True)
            print('Plotting')
           
            ind = candidates[:, 0].argsort()
            plt.plot(candidates[ind], pred_means[ind, 0], 'b-', label='Obj 1')
            plt.plot(candidates[ind], pred_means[ind, 0] - pred_vars[ind, 0], 'b--')
            plt.plot(candidates[ind], pred_means[ind, 0] + pred_vars[ind, 0], 'b--')
            plt.plot(candidates[ind], pred_means[ind, 1], 'g-', label='Obj 2')
            plt.plot(candidates[ind], pred_means[ind, 1] - pred_vars[ind, 1], 'g--')
            plt.plot(candidates[ind], pred_means[ind, 1] + pred_vars[ind, 1], 'g--')
            plt.xlabel('x')
            plt.ylabel('Objective')
            plt.title('Model predictions')
            plt.legend()
            plt.show()
        aquisition_values = aquisition_function.getAquisitionBatch(candidates,
                                                                   model,
                                                                   frontier)
        max_aquisition_index = np.argmax(aquisition_values)
        
        
        if (plots):
            plt.figure(2)
            plt.clf()
            plt.gca().grid(True)
            plt.plot(candidates[ind], aquisition_values[ind], 'r-')
            plt.xlabel('x')
            plt.ylabel('Aquisition value')
            plt.title('Aquisition function')
        
            plt.draw()
            plt.figure(3)
            ax = plt.gca()
            plt.gca().grid(True)
            ind = frontier[:, 0].argsort()
            frontier = frontier[ind, :]
            plt.plot(frontier[:, 0], frontier[:, 1], 'gs')
            flist = []
            reference_point = [10, 10]
            if ('reference' in aquisitionParams):
                reference_point = aquisitionParams['reference']
            current_point = [0, reference_point[1]]
            for i in range(0, len(frontier)):
                current_point[0] = frontier[i, 0]
                if (i > 0 or frontier[i, 1] > reference_point[1]):
                    flist.append(current_point)
                flist.append(frontier[i, :])
                current_point = current_point.copy()
                current_point[1] = frontier[i, 1]
            if (current_point[0] < reference_point[0]):
                current_point[0] = reference_point[0]
                flist.append(current_point)
            flist = np.array(flist)
            plt.plot(flist[:, 0], flist[:, 1], 'g-')    
            ax.fill_between(flist[:, 0], reference_point[1], flist[:, 1], where=reference_point[1] >= flist[:, 1], facecolor='green', alpha=0.5, hatch='//')
            plt.xlabel('Obj 1')
            plt.ylabel('Obj 2')
            plt.title('Pareto frontier')
            plt.draw()
            plt.show()

        max_aquisition_value = aquisition_values[max_aquisition_index]
        new_point = candidates[max_aquisition_index]
        init_points = np.vstack((new_point, init_points))
        new_point_value = np.reshape(np.array(f(new_point)), (1, -1))
        print('New point at {0}'.format(new_point_value))
        candidates = np.delete(candidates, max_aquisition_index, 0)

        model.addPoint(new_point, new_point_value)
        frontier = find_frontier(np.vstack((new_point_value, frontier)))

        print('Iter {0}, {1} improved to {2} in {3} time'
              .format(iter,
                      aquisition_function.getGoalName(),
                      aquisition_function.getGoalValue(frontier),
                      time.time() - iter_start)
             )
        curve.append(aquisition_function.getGoalValue(frontier))
        
        if (output_dir is not None):
            frontierfile = open('{0}/frontier.txt'.format(output_dir), 'w+')
            curvefile = open('{0}/curve.txt'.format(output_dir), 'w+')
            #pointsfile = open('{0}/points.txt'.format(output_dir), 'w+')
            #candidatesfile = open('{0}/candidates.txt'.format(output_dir), 'w+')
            for item in frontier:
                frontierfile.write("%s\n" % item)
              
            for item in np.array(curve):
                curvefile.write("%s\n" % item)
                
            np.savetxt('{0}/points.txt'.format(output_dir), init_points)
            np.savetxt('{0}/candidates.txt'.format(output_dir), candidates)
            #for item in init_points:
            #    pointsfile.write("%s\n" % item)
                
            #for item in candidates:
            #    candidatesfile.write("%s\n" % item)

    return frontier, np.array(curve)
    
def find_frontier(init_values):
    frontier_ind = []
    for i in range(0, init_values.shape[0]):
        dominated = False
        for j in range(0, init_values.shape[0]):
            if (np.all(np.all(init_values[i, :] > init_values[j, :]))):
                dominated = True

        if (not dominated):
            frontier_ind.append(i)
    
    return init_values[np.array(frontier_ind), :]
