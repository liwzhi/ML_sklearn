# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 19:35:33 2015

@author: weizhi
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import json
from sklearn.cross_validation import *
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix

test = pd.read_csv('/Users/weizhi/Downloads/DataScientistTakehome (1)/data/DS_test.csv')
train = pd.read_csv('/Users/weizhi/Downloads/DataScientistTakehome (1)/data/DS_train.csv')

# loading the data

with open('/Users/weizhi/Downloads/DataScientistTakehome (1)/data/DS_train_labels.json') as data_file:    
    data = json.load(data_file)

# get the data value
countOne = 0
countZero = 0   
 
for key in data.keys():
    if data[key] == 1:
        countOne +=1
    else:
        countZero +=1
        
label = pd.DataFrame()
label['unique_id'] = data.keys()  
label['label'] = data.values()
 
# merget the data  
result = pd.merge(label, train, on='unique_id')
train, data, label = [],[],[]

y = result['label']
del result['label']
X = result

#%% feature hashing
colToHash = ['city','state','contact_title','category','has_facebook','has_twitter']


for col in colToHash:
    result[col] = abs((result[col].apply(hash))%2**(16))

del result['unique_id']
#%% handle missing value
print ("handle missing data")
result.fillna(result.mean(),inplace=True)



#%% data preprocessing
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler, RobustScaler


standard_scaler = StandardScaler()
robust_scaler = RobustScaler()

X_train = robust_scaler.fit_transform(result)
X_train1 = standard_scaler.fit_transform(result)
#%% performace 


def performence(clf,train,label,clfName):
    re = cross_validation.ShuffleSplit(train.shape[0],n_iter=10,test_size =0.25,random_state =43)
    
    aucList = []
    accuracyList = []
    for train_index, test_index in re:
        clf.fit(train.iloc[train_index,:],y.iloc[train_index])
        pre_y = clf.predict_proba(train.iloc[test_index,:])  # probablity to get the AUC
        aucList.append(roc_auc_score(y.iloc[test_index],pre_y[:,1]))
        y_pred = clf.predict(train.iloc[test_index,:]) # get the accuracy of model 
        accuracyList.append(accuracy_score(y.iloc[test_index],y_pred))  
    print 'the classifications is ' + clfName
    print ("The AUC score is %f"%(sum(aucList)/10.))
    print ("The model accuracy is %f"%(sum(accuracyList)/10.))
    print "confusion matrix"
    print (confusion_matrix(y.iloc[test_index],y_pred))

#%% training a model 
from sklearn import ensemble
from sklearn import cross_validation
from sklearn.metrics import log_loss

from sklearn import grid_search
rf = ensemble.RandomForestClassifier(n_estimators = 200,random_state=43,class_weight="balanced",max_features=None)

parameters = {'n_estimators':[100,200,400,600],'min_samples_leaf':[2,5,10],'max_features':['sqrt',None]}
#
#svr = svm.SVC()
clf1 = grid_search.GridSearchCV(rf,parameters)
#clf.fit(result,y)
clf1.fit(result,y)

rf = clf1.best_estimator_
cv = cross_validation.cross_val_score(rf,result,y,cv=10)
print ('random forest')
print (cv.mean())
#%%
performence(rf,result,y,'gradient boosting')

performence(rf,pd.DataFrame(X_train),y,'gradient boosting')


performence(rf,pd.DataFrame(X_train1),y,'gradient boosting')

#%% 

ada = ensemble.GradientBoostingClassifier(random_state=43)

parameters = {'n_estimators':[100,200,400,600],'min_samples_leaf':[2,5,10],'max_features':['sqrt',None],'subsample':[0.5,0.8,1]}
#
#svr = svm.SVC()
clf1 = grid_search.GridSearchCV(ada,parameters)
#clf.fit(result,y)
clf1.fit(result,y)

adarf = clf1.best_estimator_
cv = cross_validation.cross_val_score(ada,result,y,cv=10)
print ('ada')
print (cv.mean())

#%%
performence(adarf,result,y,'gradient boosting')

performence(adarf,pd.DataFrame(X_train),y,'gradient boosting')

performence(adarf,pd.DataFrame(X_train1),y,'gradient boosting')




#%% classificatoin 
clf = linear_model.LogisticRegression(solver='liblinear',random_state=43)

parameters = {'penalty':['l1','l2'],'class_weight':[{0:1,1:2},{0:1,1:1}],'C':[0.01,0.1,1,10]}

clf1 = grid_search.GridSearchCV(clf,parameters)

clf1.fit(X_train,y)
lb = clf1.best_estimator_

cv = cross_validation.cross_val_score(lb,X_train,y,cv=10)
print (cv.mean())


cv = cross_validation.cross_val_score(lb,result,y,cv=10)
print (cv.mean())


cv = cross_validation.cross_val_score(lb,X_train1,y,cv=10)
print (cv.mean())
#%%
performence(lb,result,y,'logistic regression')

performence(lb,pd.DataFrame(X_train),y,'logistic regression')
performence(adarf,pd.DataFrame(X_train1),y,'gradient boosting')

#%% do the nearest neighbor 

from sklearn.neighbors import KNeighborsClassifier 

parameters = {'n_neighbors':[3,5,7,9,11,15],'weights':['uniform','distance'],'leaf_size':[10,20,30,40,50]}

Knn = KNeighborsClassifier()
clf1 = grid_search.GridSearchCV(Knn,parameters)
clf1.fit(X_train,y)

knn = clf1.best_estimator_



cv = cross_validation.cross_val_score(knn,X_train,y,cv=10)
print (cv.mean())


cv = cross_validation.cross_val_score(knn,result,y,cv=10)
print (cv.mean())


cv = cross_validation.cross_val_score(knn,X_train1,y,cv=10)
print (cv.mean())
#%%
performence(knn,result,y,'knn')
#performence(knn,pd.DataFrame(X_train),y,'knn')
#performence(knn,pd.DataFrame(X_train1),y,'knn')

#%% SVM

from sklearn import svm, grid_search
#
parameters = {'kernel':('linear','rbf','poly'),'C':[1,10,100],'gamma':[0.0001,0.01,0.1],'class_weight':[{0:1,1:1},{0:1,1:2}]}
#
svr = svm.SVC(random_state=43,probability=True)
clf = grid_search.GridSearchCV(svr,parameters)
clf.fit(X_train,y)

svr = clf.best_estimator_



cv = cross_validation.cross_val_score(svr,X_train,y,cv=10)
print (cv.mean())


cv = cross_validation.cross_val_score(svr,result,y,cv=10)
print (cv.mean())


cv = cross_validation.cross_val_score(svr,X_train1,y,cv=10)
print (cv.mean())
#%%
performence(svr,result,y,'SVM')
performence(svr,pd.DataFrame(X_train),y,'SVM')
performence(adarf,pd.DataFrame(X_train1),y,'SVM')

#%% 



#%%

import xgboost as xgb
print("Train a XGBoost model")
params = {"objective": "binary:logistic",
          "eta": 0.3,
          "max_depth": 5,
          "min_child_weight": 3,
          "silent": 1,
          "subsample": 1,
          "colsample_bytree": 0.7,
          "seed": 1}
num_trees=450
gbm = xgb.train(params, xgb.DMatrix(result, y), num_trees)


#%% validatoin 
from sklearn import cross_validation

re = cross_validation.ShuffleSplit(result.shape[0],n_iter=10,test_size =0.25,random_state =43)

aucList = []
for train_index, test_index in re:
    gbm = xgb.train(params,xgb.DMatrix(result.iloc[train_index,:],y.iloc[train_index]),num_trees)
    pre_y = gbm.predict(xgb.DMatrix(result.iloc[test_index]))
    aucList.append(roc_auc_score(y.iloc[test_index],pre_y))

print "xbgboost"
print sum(aucList)/10.



#%% Deep Learning
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
 
    
layers0 = [('input', InputLayer),
           ('dense0', DenseLayer),
           ('dropout', DropoutLayer),
           ('dense1', DenseLayer),
           ('dense2',DenseLayer),
           ('output', DenseLayer)]

net0 = NeuralNet(layers=layers0,
                 
                 input_shape=(None, 13),
                 dense0_num_units=100,
                 dropout_p=0.5,
                 dense1_num_units=200,
                 dense2_num_units=100,
                 output_num_units=2,
                 output_nonlinearity=softmax,
                 
                 update=nesterov_momentum,
                 update_learning_rate=0.02,
                 update_momentum=0.9,
                 
                 eval_size=0.2,
                 verbose=1,
                 max_epochs=500)    
#%%
netAccuracy = []             
                
                 
encoder = LabelEncoder() 

#%% ensemble all the models -- find the weights
#% log loss function
from scipy.optimize import minimize

def log_loss_func(weights):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = 0
    for weight, prediction in zip(weights, predictions):
            final_prediction += weight*prediction

    return log_loss(y.iloc[test_index], final_prediction)
    
re = cross_validation.ShuffleSplit(result.shape[0],n_iter=10,test_size =0.25,random_state =43)

cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
bounds = [(0,1)]*6

weights = []
for train_index, test_index in re:
    aucList = []
    predictions = []
    gbm = xgb.train(params,xgb.DMatrix(result.iloc[train_index,:],y.iloc[train_index]),num_trees)
    rf.fit(X_train[train_index,:],y[train_index])
    ada.fit(X_train[train_index,:],y[train_index])
    svr.fit(X_train[train_index,:],y[train_index])
    knn.fit(X_train[train_index,:],y[train_index])
    
    y_train = encoder.fit_transform(y[train_index]).astype(np.int32)

    net0.fit(X_train[train_index,:],y_train)
    predictions.append(svr.predict_proba(X_train[test_index])[:,1])
    predictions.append(gbm.predict(xgb.DMatrix(result.iloc[test_index])))
    predictions.append(ada.predict_proba(X_train[test_index])[:,1])
    predictions.append(rf.predict_proba(X_train[test_index])[:,1])
    predictions.append(knn.predict_proba(X_train[test_index])[:,1])
    predictions.append(net0.predict_proba(X_train[test_index])[:,1])
    netAccuracy.append(roc_auc_score(y.iloc[test_index],net0.predict_proba(X_train[test_index])[:,1]))
    starting_values = [1./6]*len(predictions)

  #  train_eval_probs = 0.30*svr.predict_proba(X_train[test_index])[:,1] + 0.15*gbm.predict(xgb.DMatrix(result.iloc[test_index])) \
   #    + 0.15*ada.predict_proba(X_train[test_index])[:,1] + 0.35*rf.predict_proba(X_train[test_index])[:,1] \
    #   +  0.05*knn.predict_proba(X_train[test_index])[:,1]
    res = minimize(log_loss_func, starting_values, method='SLSQP', bounds=bounds, constraints=cons)
    print('Ensamble Score: {best_score}'.format(best_score=res['fun']))
    print('Best Weights: {weights}'.format(weights=res['x']))
    weights.append(res['x'])
  #  aucList.append(roc_auc_score(y.iloc[test_index],train_eval_probs))

print "Deep Learning - CNN"
print sum(netAccuracy)/10.

#%% get the final model



weight = np.zeros(6)
for i in range(6):
    weight += weights[i]
weight = weight/6.
aucList = []
for train_index, test_index in re:

    gbm = xgb.train(params,xgb.DMatrix(result.iloc[train_index,:],y.iloc[train_index]),num_trees)
    rf.fit(X_train[train_index,:],y[train_index])
    ada.fit(X_train[train_index,:],y[train_index])
    svr.fit(X_train[train_index,:],y[train_index])
    knn.fit(X_train[train_index,:],y[train_index])
    y_train = encoder.fit_transform(y[train_index]).astype(np.int32)

    net0.fit(X_train[train_index,:],y_train)


    train_eval_probs = weight[0]*svr.predict_proba(X_train[test_index])[:,1] + weight[1]*gbm.predict(xgb.DMatrix(result.iloc[test_index])) \
      + weight[2]*ada.predict_proba(X_train[test_index])[:,1] + weight[3]*rf.predict_proba(X_train[test_index])[:,1] \
       +  weight[4]*knn.predict_proba(X_train[test_index])[:,1] + weight[5]*net0.predict_proba(X_train[test_index])[:,1]
    aucList.append(roc_auc_score(y.iloc[test_index],train_eval_probs))

print "ensemble learning"
print sum(aucList)/10.
#%% generate the outputs 


gbm = xgb.train(params,xgb.DMatrix(X_train,y),num_trees)
rf.fit(X_train,y)
ada.fit(X_train,y)
svr.fit(X_train,y)
knn.fit(X_train,y)
y_train = encoder.fit_transform(y).astype(np.int32)

net0.fit(X_train,y_train)
#%% test data 
colToHash = ['city','state','contact_title','category','has_facebook','has_twitter']


for col in colToHash:
    test[col] = abs((test[col].apply(hash))%2**(16))

ID = test['unique_id']
del test['unique_id']
#%% handle missing value
print ("handle missing data")
test.fillna(test.mean(),inplace=True)



#%% data preprocessing



X_test = robust_scaler.fit_transform(test)
X_test1 = standard_scaler.fit_transform(test)

#%% final model 

Finalmodel = weight[0]*svr.predict_proba(X_test)[:,1] + weight[1]*gbm.predict(xgb.DMatrix(X_test)) \
  + weight[2]*ada.predict_proba(X_test)[:,1] + weight[3]*rf.predict_proba(X_test)[:,1] \
   +  weight[4]*knn.predict_proba(X_test)[:,1] + weight[5]*net0.predict_proba(X_test)[:,1]

#%% generate the outputs
submit = pd.DataFrame()
submit['ID'] = ID
submit['prob'] = Finalmodel
submit.to_csv('/Users/weizhi/Downloads/DataScientistTakehome (1)/radius_data_test.csv',index = False)























    
