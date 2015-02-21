# -*- coding: utf-8 -*-
"""
Created on Mon Feb 02 21:51:28 2015

@author: Weizhi
"""

import glob
import os
import pandas as pd
import featureExtraction as fx
from scipy.io import wavfile
import time 
start = time.time()
pathTrain = 'C:/Users/Weizhi/Desktop/QeexoInterview-BasicFingerSenseClassifier/train/train'
pathTest = 'C:/Users/Weizhi/Desktop/QeexoInterview-BasicFingerSenseClassifier/test/test'
# get the path folder
# get the path folder
dataFeature = fx.featuerExtraction()
def trainTest(path):
    folderName = os.listdir(path)
    pathName = []
    for item in folderName:
        pathName.append(os.path.join(path,item))
    return pathName

#%% from 1 start, to avoid the .DS_Store for one user pathName[1]
def getUserData(pathName,userIndex):
    trainData = os.listdir(pathName[userIndex])
    count = 1
    pathNameTrain = []
    for item in trainData:
        pathNameTrain.append(os.path.join(pathName[userIndex],item))
    
    # save the labels, and training features for one user
           
    label = []
    dataGet = None
    trainSingle = glob.glob(pathNameTrain[0] + '/*.csv')
    dataWav = glob.glob(pathNameTrain[0] + '/*.wav')
    if len(trainSingle) ==0:
        trainSingle = glob.glob(pathNameTrain[1] + '/*.csv') 
        count +=1
        trainData = trainData[1:]
#            print pathNameTrain[0]
    Name = pathNameTrain[0].split('-')
    label.append(Name[-1])
    dataGet = pd.read_csv(trainSingle[0])
    
    if len(dataWav) ==0:
        dataWav = glob.glob(pathNameTrain[1] + '/*.wav') 
        
    dataWav = wavfile.read(dataWav[0])[1]
#    data = pd.DataFrame(dataWav.tolist())
    #%% get the time series features
#    output = pd.DataFrame()
    Output = dataFeature.featureExtract(dataWav).T
    for index in range(count,len(pathNameTrain)):
        Name = pathNameTrain[index].split('-')
        label.append(Name[-1])
        trainSingle = glob.glob(pathNameTrain[index] + '/*.csv')
        data = pd.read_csv(trainSingle[0])
        dataGet = dataGet.append(data)
        #%% time series data extract
        dataWav = glob.glob(pathNameTrain[index] + '/*.wav')           
        dataWav = wavfile.read(dataWav[0])[1].T
        OutputData = dataFeature.featureExtract(dataWav).T
        Output = Output.append(OutputData)
    dataGet['label'] = label
    dataGet['index'] = trainData  
    dataGet = pd.concat((dataGet,Output),axis=1)
    return dataGet

#%% to different users, star from one, train, 
pathNameTraining = trainTest(pathTrain)
dataTrain = getUserData(pathNameTraining,1)


# get all the training features
for user in range(2,len(pathNameTraining)):
    print (user)
    output1 = getUserData(pathNameTraining,user)
    dataTrain = dataTrain.append(output1)
    print dataTrain.shape
# ouble check the data
knuckleTrain = dataTrain[dataTrain['label']=='knuckle']
padTrain = dataTrain[dataTrain['label']=='pad']  
#should contain 20,659 training instances (10,255 knuckle, 10404 pad)
print (knuckleTrain.shape)
print (padTrain.shape)
#%% to different users, star from one test
pathNameTesting = trainTest(pathTest)
dataTest = getUserData(pathNameTesting,1)
# get all the training features
for user in range(2,len(pathNameTesting)):
    print(user)
    output1 = getUserData(pathNameTesting,user)
    dataTest = dataTest.append(output1)  
# Your test.zip file should contain 10,528 test instances (5,263 knuckle, 5,265 pad)
print (dataTest.shape) 
#%%     
dataTrain.to_csv('E:/BasicFingerSenseClassifier/TrainfeaturesT')
dataTest.to_csv('E:/BasicFingerSenseClassifier/TestfeaturesT')     
end = time.time()
#%% reading the data from dictionary
dataTrain = pd.read_csv('C:/Users/Weizhi/Desktop/QeexoInterview-BasicFingerSenseClassifier/TrainfeaturesT')
dataTest = pd.read_csv('C:/Users/Weizhi/Desktop/QeexoInterview-BasicFingerSenseClassifier/TestfeaturesT')
featureTime = end - start
print "Time to extract feature", featureTime, "second"

#%% come to machine learning part:
from sklearn.cross_validation import KFold
from operator import itemgetter 
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
#%%
knuckleTrain = dataTrain[dataTrain['label']=='knuckle']
padTrain = dataTrain[dataTrain['label']=='pad']  
#%% for Knuckle
XX = knuckleTrain.iloc[:,1:7]
YY = knuckleTrain.iloc[:,9:]
X = pd.concat((XX,YY),axis=1)

X.hist()
plt.suptitle('the knuckle')

y = dataTrain['label']

meanValue = X.mean()
stdValue = X.std()
count = 0
sampleIndex = []
for i in range(len(X)):
    count = 0
    currInstance = X.iloc[i,:]    
    for j in range(len(currInstance)):
        left = meanValue[j] -2*stdValue[j]
        right = meanValue[j] + 2*stdValue[j]
        if left<currInstance[j] and right>currInstance[j]:
            count +=1
    if count>=20:
        sampleIndex.append(i)
print 'the number of sample is %d' % len(sampleIndex)
print 'the number of outliers is %d' % (len(X) - len(sampleIndex))

newknuckleTrain = X.iloc[sampleIndex,:]



#%%

#for opad

XX = padTrain.iloc[:,1:7]
YY = padTrain.iloc[:,9:]
X = pd.concat((XX,YY),axis=1)
y = dataTrain['label']

X.hist()
plt.suptitle('the pad')
meanValue = X.mean()
stdValue = X.std()
count = 0
sampleIndex = []
for i in range(len(X)):
    count = 0
    currInstance = X.iloc[i,:]    
    for j in range(len(currInstance)):
        left = meanValue[j] -2*stdValue[j]
        right = meanValue[j] + 2*stdValue[j]
        if left<currInstance[j] and right>currInstance[j]:
            count +=1
    if count>=20:
        sampleIndex.append(i)
print 'the number of sample is %d' % len(sampleIndex)
print 'the number of outliers is %d' % (len(X) - len(sampleIndex))


newPadTrain = X.iloc[sampleIndex,:]

X = newknuckleTrain.append(newPadTrain)


newknuckleTrain.hist()
plt.suptitle('the knuckle after remove outliers')
# 1 is for kunck, 0 for pad
y_label = [1]*len(newknuckleTrain) + [0]*len(newPadTrain)

X_train, X_test, y_train, y_test = train_test_split(
        X,y_label,test_size=0.5,random_state=0)
#%% random forest 
tuned_parameters= [{'n_estimators':[100,200,300],'min_samples_split':[100,150,200,250,300]}]     
clf = RandomForestClassifier(n_estimators=100, max_depth=None,
     min_samples_split=100, random_state=0)
clf_Search = GridSearchCV(clf,tuned_parameters,cv=5,scoring='precision')
clf_Search.fit(X_train,y_train)
#%% get the best random forest classification

clf = clf_Search.best_estimator_

#clf.fit(X_train,y_train)
#%% get the feature importance
clf.feature_importances_
#%%
import numpy as np
importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
length = X.shape[1]
for f in range(length):
    print("%d. feature %d %s (%f)" % (f + 1, indices[f], X.keys()[indices[f]], importances[indices[f]]))

LabelName = ['minor','major', 'y','x','orientation','pressure']
name = []
for i in range(length):
    name.append(X.keys()[indices[i]][-5:])
# Plot the feature importances of the forest
import pylab as plt
plt.figure()
plt.title("Feature importances")
plt.bar(range(length), importances[indices],
       color="r",  align="center")
plt.xticks(range(length), name)
plt.xlim([-1, length])
plt.legend()
plt.show()

print clf
#%% do another cross validation:


kf = KFold(len(X_test), n_folds=10)
f1 = []
for train, test in kf:
    clf.fit(X_test[train,:],itemgetter(*train)(y_test))
    y_testResult = clf.predict(X_test[test,:])
    f1.append(f1_score(itemgetter(*test)(y_test),y_testResult,average='macro'))
     
print 'random forest accuracy'
print f1      

##y_testResult = clf.predict(X_test)
##print f1_score(y_test,y_testResult,average='macro')
#
##clf = GridSearchCV(svm.SVC(C=1),tuned_parameters,cv=5,scoring='precision')
###%% SVM for the classifcation
##tuned_parameters = [{'kernel':['rbf'], 'gamma':[1e-3,1e-4],
##                     'C':[1,10,100,1000]},
##                    {'kernel':['linear'],'C':[1,10,100,1000]}]
##
#clf_SVM = svm.SVC()
#clf_SVM.fit(X_train,y_train)
#
#best = clf_SVM.predict(X_test)
#f1_score(best, y_test, average='macro')
#f1_score(itemgetter(*test)(y_test),y_testResult,average='macro')
#
#kf = KFold(len(X_test), n_folds=5)
#f2 = []
#for train, test in kf:
#    clf_SVM.fit(X_test[train,:],itemgetter(*train)(y_test))    
#    y_testResult = clf_SVM.predict(X_test[test,:])
#    f2.append(f1_score(itemgetter(*test)(y_test),y_testResult,average='macro'))
#print 'SVM accuracy'
#print f2     
#%% K-NN classification 
from sklearn import neighbors
clf_knn = neighbors.KNeighborsClassifier(n_neighbors=3, weights='uniform')
clf_knn.fit(X_train,y_train)
kf = KFold(len(X_test), n_folds=10)
f3 = []
for train, test in kf:
    clf_knn.fit(X_test[train,:],itemgetter(*train)(y_test))
    y_testResult = clf_knn.predict(X_test[test,:])
    f3.append(f1_score(itemgetter(*test)(y_test),y_testResult,average='macro'))
    
print 'KNN accuracy'
print f3   

#%% GBD
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train,y_train)
kf = KFold(len(X_test), n_folds=10)
f4 = []
for train, test in kf:
    gnb.fit(X_test[train,:],itemgetter(*train)(y_test))
    y_testResult = gnb.predict(X_test[test,:])
    f4.append(f1_score(itemgetter(*test)(y_test),y_testResult,average='macro'))
print 'GNB accuracy'
print f4
#%% get the logisticregression
from sklearn.linear_model import LogisticRegression
C = 10
clf_l1_LR = LogisticRegression(C=C, penalty='l1', tol=0.001)

clf_l1_LR.fit(X_train,y_train)
kf = KFold(len(X_test), n_folds=10)
f5 = []
for train, test in kf:
    clf_l1_LR.fit(X_test[train,:],itemgetter(*train)(y_test))
    y_testResult = clf_l1_LR.predict(X_test[test,:])
    f5.append(f1_score(itemgetter(*test)(y_test),y_testResult,average='macro'))
print 'clf_l1_LR accuracy'
print f5

#%% classifcation unknow data
XX_test = dataTest.iloc[:,1:7]
YY_test = dataTest.iloc[:,9:]
X_test = pd.concat((XX_test,YY_test),axis=1)

#X_test = dataTest.iloc[:,a[0]]
#X_test = dataTest.iloc[:,0:6]
clf = clf_Search.best_estimator_
clf = clf_l1_LR
dataTest['label'] = clf.predict(X_test)
y_label = []
# 1 is for kunck, 0 for pad
for item in dataTest['label']:
    if item==0:
        y_label.append('pad')
    else:
        y_label.append('knuckle')
saveOutput = pd.DataFrame()
saveOutput['timestamp'] = dataTest['index']
saveOutput['label'] = y_label

knuckleTrain = saveOutput[saveOutput['label']=='knuckle']
padTrain = saveOutput[saveOutput['label']=='pad']  
#should contain 20,659 training instances (10,255 knuckle, 10404 pad)
print 'the number of test output'
print 'the test of knuckle number is %d' %(knuckleTrain.shape[0])
print 'the test of pad number is %d' %(padTrain.shape[0])

saveOutput.to_csv('C:/Users/Weizhi/Desktop/QeexoInterview-BasicFingerSenseClassifier/Output_Li') 
endEnd = time.time()

classifcation = endEnd - end
print "Time to extract feature", classifcation, "second"






