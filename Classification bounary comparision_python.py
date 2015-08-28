# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 15:08:59 2015

@author: weizhi
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 11:58:20 2015

@author: weizhi
"""



#%% Plot, run this part at first 


from matplotlib.colors import ListedColormap
from sklearn.metrics import roc_curve, auc
import numpy as np
from matplotlib import pyplot as plt

from sklearn.preprocessing import LabelEncoder
from scipy import interp

encoder = LabelEncoder()

# utility function to plot the decision surface
def plot_surface(est, x_1, x_2, ax=None, threshold=0.0, contourf=False):
    from matplotlib import pyplot as plt

    """Plots the decision surface of ``est`` on features ``x1`` and ``x2``. """
    xx1, xx2 = np.meshgrid(np.linspace(x_1.min(), x_1.max(), 100), 
                           np.linspace(x_2.min(), x_2.max(), 100))
    # plot the hyperplane by evaluating the parameters on the grid
    X_pred = np.c_[xx1.ravel(), xx2.ravel()]  # convert 2d grid into seq of points
   # if hasattr(est, 'predict_proba'):  # check if ``est`` supports probabilities
        # take probability of positive class
      #  pred = est.predict_proba(X_pred)[:, 1]
    #else:
    pred = est.predict(X_pred)
    Z = pred.reshape((100, 100))  # reshape seq to grid
    if ax is None:
        ax = plt.gca()
    # plot line via contour plot
    
    if contourf:
        ax.contourf(xx1, xx2, Z, levels=np.linspace(0, 1.0, 10), cmap=plt.cm.RdBu, alpha=0.6)
    ax.contour(xx1, xx2, Z, levels=[threshold], colors='black')
    ax.set_xlim((x_1.min(), x_1.max()))
    ax.set_ylim((x_2.min(), x_2.max()))
    
    
def plot_datasets(clf_name, est=None):
   # from matplotlib import pyplot as plt

    """Plotsthe decision surface of ``est`` on each of the three datasets. """
    fig, axes = plt.subplots( 1,3, figsize=(10, 4))
    
    for (name, ds), ax in zip(datasets.iteritems(), axes):
        X_train = ds['X_train']
        y_train = ds['y_train']
        X_test = ds['X_test']
        y_test = ds['y_test']
        
        # plot test lighter than training
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
        # plot limits
        ax.set_xlim(X_train[:, 0].min(), X_train[:, 0].max())
        ax.set_ylim(X_train[:, 1].min(), X_train[:, 1].max())
        # no ticks
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_ylabel('$x_1$')
        ax.set_xlabel('$x_0$')
        if len(clf_name[0]) !=1: # is 'raw data'
            ax.set_title(clf_name[0] )
        else:
            ax.set_title(name)

            
        err = 0
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        all_tpr = []
        DL = 'Deep Learning'
        if clf_name[0] == DL:
            y_train = encoder.fit_transform(y_train).astype(np.int32)
            y_test = encoder.fit_transform(y_test).astype(np.int32)


        
        
        if est is not None:
            est.fit(X_train, y_train)
            plot_surface(est, X_train[:, 0], X_train[:, 1], ax=ax, threshold=0.5, contourf=False)
            err = (y_test != est.predict(X_test)).mean() 
            if hasattr(est, 'predict_proba'):
                # check if ``est`` supports probabilities
                probas_ = est.predict_proba(X_test)
                fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
                mean_tpr += interp(mean_fpr, fpr, tpr)
                mean_tpr[0] = 0.0
                err = auc(fpr, tpr)
          #  else:
               # err = (y_test != est.predict(X_test)).mean() 

            ax.text(0.88, 0.02, '%.2f' % err, transform=ax.transAxes)
   # return fig
            
    fig.subplots_adjust(left=.02, right=.98)

#%% generate the dataset
#import PlotShow as Plot

import numpy as np
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification,make_s_curve
from sklearn.cross_validation import train_test_split

# generate 3 synthetic datasets
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = OrderedDict()
for name, (X, y) in [('moon', make_moons(noise=0.3, random_state=0,n_samples=200)),
                    ('circles', make_circles(noise=0.2, factor=0.5, random_state=1,n_samples=200)),
                    ('linear', linearly_separable)]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=1)
    # standardize data
    scaler = StandardScaler().fit(X_train)
    datasets[name] = {'X_train': scaler.transform(X_train), 'y_train': y_train,
                      'X_test': scaler.transform(X_test), 'y_test': y_test}
    



import matplotlib.gridspec as gridspec
plot_datasets([' '])


fig = plt.figure()

ax1 = fig.add_subplot(211)

#%% GBM 
#fig, axes = plt.subplots(1, 3, figsize=(10, 4))

from sklearn import ensemble
clf = ensemble.GradientBoostingClassifier(n_estimators = 50,max_depth = 5)

plot_datasets(['GBM  ntress=50, max_depth=5'],clf)


#%% Random forest

fig = plt.figure()
est = ensemble.RandomForestClassifier(n_estimators =10,max_depth=20)
plot_datasets(['DRF ntrees=10, max_depth=20'],est) 



#%%  Logistici Regaression
from sklearn.linear_model import LogisticRegression
est = LogisticRegression()
plot_datasets(['Logistic regression'],est)


#%% GLM -- SGD 

from sklearn import linear_model
clf = linear_model.SGDClassifier()
plot_datasets(['GLM'],clf)

#%% Deep Learning
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
 
    
layers0 = [('input', InputLayer),
           ('dense0', DenseLayer),
          # ('dropout', DropoutLayer),
           ('dense1', DenseLayer),
           ('dense2',DenseLayer),
           ('output', DenseLayer)]

net0 = NeuralNet(layers=layers0,
                 
                 input_shape=(None, 2),
                 dense0_num_units=50,
               #  dropout_p=0.5,
                 dense1_num_units=50,
                 dense2_num_units=50,
                 output_num_units=2,
                 output_nonlinearity=softmax,
                 
                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,
                 
                 eval_size=0.2,
                 verbose=1,
                 max_epochs=1000)
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train).astype(np.int32)


#plot_datasets(net0)

net0.fit(X_train,y_train)




plot_datasets(['Deep Learning'],net0)




