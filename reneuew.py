# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 16:23:27 2015

@author: Algorithm 001
"""

from sklearn import linear_model
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
pd.__version__
train = pd.read_csv('D:\\Data\\Restruant revenue prediction/train.csv')
test = pd.read_csv('D:\\Data\\Restruant revenue prediction/test.csv')
sample = pd.read_csv('D:/Data/Restruant revenue prediction/sampleSubmission.csv')

y = train.iloc[:,-1].as_matrix()
X_pre= train.iloc[:,5:-1]
Xtest = test.iloc[:,5:]


plt.figure()
plt.hist(y)
plt.show()

plt.figure()
plt.hist(np.log(y))
plt.show()
plt.title("after log data")




#%% prepossing the data
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


from sklearn import preprocessing


Best = SelectKBest(f_regression, k=20).fit(X_pre, y)
X = Best.transform(X_pre)
indexFeature1 = Best.get_support(indices=True)




X = preprocessing.PolynomialFeatures(interaction_only=True,include_bias=False).fit_transform(X)


Best = SelectKBest(f_regression, k=200).fit(X, y)
X = Best.transform(X)
indexFeature2 = Best.get_support(indices=True)
#Xtest = preprocessing.PolynomialFeatures().fit_transform(Xtest)
y = np.log(y)

#%% to test data

Xtest = Xtest.iloc[:,indexFeature1]

Xtest = preprocessing.PolynomialFeatures(interaction_only=True,include_bias=False).fit_transform(Xtest)
#Xtest = preprocessing.PolynomialFeatures(interaction_only=True,include_bias=False).fit_transform(Xtest)
Xtest = Xtest[:,indexFeature2]

#%% feauter understanding






# come to classication
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
clf_svr = svm.SVR()

clf_RF = RandomForestRegressor(n_estimators=300,criterion = 'mse')

#eavluation classication

from sklearn import cross_validation
from sklearn.metrics import mean_absolute_error
def loss(clf,X,y):
    error = pd.Series()
    
    loo = cross_validation.LeaveOneOut(X.shape[0])
    count = 0
    for train, test in loo:
        clf.fit(X[train,:],y[train])
        y_predict = clf.predict(X[test,:])
        error.loc[count] = mean_absolute_error(np.exp(y[test]), np.exp(y_predict))
        count +=1
    print 'the mean error of the data is %d' % error.mean()
    return error

error_SVR = loss(clf_svr,X,y)
error_RF = loss(clf_RF,X,y)

#%% submit
clf =clf_svr.fit(X,y)

y_submit =  np.exp(clf.predict(Xtest))
#%%
submit = pd.read_csv('D:/Data/Restruant revenue prediction/sampleSubmission.csv')

submit.iloc[:,1:] = y_submit
submit['Id'] = submit['Id'].astype(int)
submit.to_csv('D:/Data/Restruant revenue prediction/sampleSubmissionTran.csv',index = False)



#%% bias 
from sklearn.learning_curve import learning_curve
train_size, train_scores, valid_scores = learning_curve(clf_svr,X,y,train_sizes=[50,80,109],cv=5)




def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt



















