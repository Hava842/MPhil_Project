from models.DeepGP.GPNetwork import GPNetwork

import numpy as np
import pylab
import matplotlib.pyplot as plt

range_ = 1.0

def modelplot(model, node, xx, yy, name):
    zz = pylab.zeros(xx.shape)
    data = []
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            data.append([xx[i,j], yy[i,j]])
    data = np.array(data)
    model.session.run(model.set_for_training.assign(0.0))
    fd = {node.input_means:data,
          node.input_vars: np.zeros_like(data)}
    pred, inducing = model.session.run((node.output_means, node.z),
                                        feed_dict=fd)
    k = 0
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            zz[i, j] = pred[k, 0]
            k += 1

    pylab.figure()
    pylab.pcolor(xx,yy,zz, cmap='RdBu', vmin=-range_, vmax=range_)
    pylab.colorbar()
    plt.scatter(inducing[:, 0], inducing[:, 1], c='k')
    plt.xlim(-range_, range_)
    plt.ylim(-range_, range_)
    plt.savefig(r'figs/{0}.eps'.format(name))
    plt.savefig(r'figs/{0}.pdf'.format(name))
    plt.savefig(r'figs/{0}.png'.format(name))

def prettyplot(f):
    xx, yy = pylab.meshgrid(pylab.linspace(-range_,range_, 100),
                            pylab.linspace(-range_,range_, 100))
    zz = pylab.zeros(xx.shape)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            zz[i,j] = f(np.array([xx[i,j], yy[i,j]]))
    pylab.pcolor(xx,yy,zz, cmap='RdBu', vmin=-range_, vmax=range_)
    pylab.colorbar()
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title('Training data')
    plt.savefig('figs/orig_data.eps')
    plt.savefig('figs/orig_data.pdf')
    plt.savefig('figs/orig_data.png')

    trainx, trainy = pylab.meshgrid(pylab.linspace(-1,1, 25),
                                    pylab.linspace(-1,1, 25))

    modelParams = {'model':'dgp',
                   'maxiter': 300,
                   'minibatch_size': 300,
                   'layer_types': ['gp', 'noisy', 'gp', 'noisy'],
                   'layer_nodes': [2, 1, 2, 1],
                   'early_stopping': False}

    training_data = []
    training_targets = []
    for i in range(trainx.shape[0]):
        for j in range(trainx.shape[1]):
            training_data.append([trainx[i, j], trainy[i, j]])
            training_targets.append(f(np.array([trainx[i,j], trainy[i,j]])).flatten())

    
    model = GPNetwork(np.array(training_data), np.array(training_targets), modelParams)

    zz = pylab.zeros(xx.shape)
    data = []
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            data.append([xx[i,j], yy[i,j]])
    data = np.array(data)
    model.session.run(model.set_for_training.assign(0.0))
    fd = {model.data_placeholder: data}
    pred = model.session.run((model.output_mean),
                                        feed_dict=fd)
    k = 0
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            zz[i, j] = pred[k, 0]
            k += 1

    pylab.figure()
    pylab.pcolor(xx,yy,zz, cmap='RdBu', vmin=-range_, vmax=range_)
    pylab.colorbar()
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.title('DGP model')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.savefig(r'figs/dgp_model.eps')
    plt.savefig(r'figs/dgp_model.pdf')
    plt.savefig(r'figs/dgp_model.png')

    for i, k in [(1, 0), (1, 1), (3, 0)]:
        modelplot(model, model.layers[i].nodes[k], xx, yy, 'layer{0}node{1}'.format(i, k))
