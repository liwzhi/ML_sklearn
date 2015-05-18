# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 00:24:46 2015

@author: Algorithm 001
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 19:55:10 2015

@author: Weizhi
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal

trainData = pd.read_csv('C:/Users/Weizhi/Desktop/Data-otto/train.csv')
testData = pd.read_csv('C:/Users/Weizhi/Desktop/Data-otto/test.csv')

trainData = pd.DataFrame((np.random.permutation(trainData)))





#%% loading the data
X = trainData.iloc[:,1:-1]
yy = trainData.iloc[:,-1]


import sys
sys.path.append('C:/Users/Weizhi/Downloads/xgboost-master (2)/xgboost-master/wrapper')

# insert path to wrapper above
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
import xgboost as xg


param = {"objective" : "multi:softprob",
"eval_metric" : "mlogloss",
"num_class" : 9,
"gamma" : 0,
"nthread" : 8,
"eta" : 0.05,
"max_depth" : 12,
"min_child_weight" : 4,
"subsample" : .9,
"colsample_bytree" : .8}

xgb_model = xg.XGBClassifier(learning_rate=0.1, n_estimators=100, silent=True, objective="multi:softprob",
                 nthread=-1, max_delta_step=0, subsample=.9, colsample_bytree=.8,
                 base_score=0.5, seed=0)

clf = GridSearchCV(xgb_model,
                   {'max_depth': [6,8,10,12,14],
                    'n_estimators': [100,200,250],'gamma':[0.1,0,1],'min_child_weight':[4,6,8]}, verbose=1,cv=10)

clf.fit(X.values,yy.values)

clf.best_estimator_

#XGBClassifier(base_score=0.5, colsample_bytree=1, gamma=0, learning_rate=0.1,
#       max_delta_step=0, max_depth=8, min_child_weight=1, n_estimators=200,
#       nthread=-1, objective='multi:softprob', seed=0, silent=True,
#       subsample=1)


sample = clf.predict_proba(testData.iloc[:,1:])








#%%
X.describe()

#%% seperate three different parts data
index1 = X.shape[0]/3
index2 = 2*(X.shape[0]/3)


dataOne = trainData.iloc[:index1,1:]
dataSecond = trainData.iloc[(index1+1):index2,1:]
dataThird = trainData.iloc[(index2+1):,1:]


Data = dataOne.groupby([94]).groups




ClassOne = dataOne.iloc[Data['Class_1'],:-1]


OneFeature = ClassOne.iloc[:,3]




plt.figure()
plt.hist(OneFeature)

#%% using the random forest to reduce the dimension
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2,1e-3, 1e-4,1e-5],
                     'C': [1, 10, 100, 1000]}]

clf = GridSearchCV(SVC(C=1), tuned_parameters)
clf.fit(dataOne.iloc[:,:-1], dataOne.iloc[:,-1])
#%% set 2 and set 3

clf2 = GridSearchCV(SVC(C=1), tuned_parameters)
clf2.fit(dataSecond.iloc[:,:-1], dataSecond.iloc[:,-1])


clf3 = GridSearchCV(SVC(C=1), tuned_parameters)
clf3.fit(dataThird.iloc[:,:-1], dataThird.iloc[:,-1])

clf.best_estimator_
clf2.best_estimator_
clf3.best_estimator_
#%%

clf1 = SVC(C=10, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.001,
  kernel='rbf', max_iter=-1, probability=True, random_state=None,
  shrinking=True, tol=0.001, verbose=False)
  
clf1.fit(dataOne.iloc[:,:-1], dataOne.iloc[:,-1]) 


clf2 = SVC(C=10, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.001,
  kernel='rbf', max_iter=-1, probability=True, random_state=None,
  shrinking=True, tol=0.001, verbose=False)

clf2.fit(dataOne.iloc[:,:-1], dataOne.iloc[:,-1]) 

clf3 = SVC(C=10, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.001,
  kernel='rbf', max_iter=-1, probability=True, random_state=None,
  shrinking=True, tol=0.001, verbose=False)

clf3.fit(dataOne.iloc[:,:-1], dataOne.iloc[:,-1]) 








sample1 = clf1.predict_proba(testData.iloc[:,1:])

sample2 = clf2.predict_proba(testData.iloc[:,1:])

sample3 = clf3.predict_proba(testData.iloc[:,1:])



XGbsample = pd.read_csv('C:/Users/Weizhi/Desktop/Data-otto/sample_submission_RooboostTree.csv')


average = np.mean([sample1,sample2,sample3,XGbsample.iloc[:,1:]],axis=0)







#%% clustering to evaluate the data set
from sklearn.cluster import KMeans

kmeans = KMeans(init='k-means++',n_clusters=9,n_init=9)
kmeans.fit(dataOne.iloc[:,:-1])


labels = kmeans.labels_

number = []
groundTruth = []
for i in range(9):
    number.append(len(filter(lambda y:y==i,labels)))
    key = Data.keys()[i]
    groundTruth.append(len(Data[key]))


#%% plot the image
import matplotlib.cm as cm

colors = cm.rainbow(np.linspace(0,1,100))

plt.figure()
for i in range(100):
    plt.plot(ClassOne.iloc[i,:],color = colors[i,:])




























#%% data clearn 
from pandas.tools.plotting import andrews_curves








import numpy as np
import pylab as plt
data = np.corr(dataOne)

NaNs = np.isnan(data)
data[NaNs] = 0

plt.figure()
plt.imshow(dataOne)
plt.colorbar()






#%%
arr = np.arange(9).reshape((3, 3))

np.random.shuffle(arr)

np.random.permutation(arr)










#%%
Testdata = testData.iloc[:,1:]
label = []
for i in range(1,10):
    label.append('Class' + '_' +  str(i))
#%% valiation the data
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
def loss(clf,X,y):
    best = 0
    kf = KFold(X.shape[0],n_folds = 5,shuffle =True)    
    scores = []
    for train, test in kf:
        clf.fit(X.iloc[train,:],y[train])
        y_pred=clf.predict(X[test])
        
        scores.append(accuracy_score(y[test],y_pred))
        if best<accuracy_score(y[test],y_pred):
            clf_best = clf
            best = accuracy_score(y[test],y_pred)
#        confusion_matrix(y[test], y_pred)
    print scores
    mean =0
    for i in range(len(scores)):
        mean +=scores[i]
    print 'the mean score is %f' % mean/10.
    return clf_best,scores    




#%% select this data 
y = []
for i in range(len(yy)):
    curr = yy[i].split('_')
    y.append(int(curr[1]))



from sklearn.cross_validation import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)

#%% build the neural network
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
import random
from pybrain.datasets import ClassificationDataSet

net = buildNetwork(93,500,9,outclass=SoftmaxLayer,bias=True)
trainer = BackpropTrainer(net, momentum=0.1, verbose=True, weightdecay=0.01, learningrate=0.01)

all_inds = range(X.shape[0])

for i in range(20):
    random.shuffle(all_inds)
    # split the indexs into lists with the indices for each batch
    batch_inds = [all_inds[i:i+1000] for i in range(0,len(all_inds),5000)]

    # train each batch
    for inds in batch_inds:
        # rebuild the dataset
        ds = ClassificationDataSet(93,nb_classes=9)
        for index in range(len(inds)):
            print index
            x_i = X.iloc[index,:].values.tolist()
            y_i = y[index]
#        for x_i, y_i in zip(X.iloc[inds,:].values.tolist(),y[inds]):
            ds.appendLinked(x_i, y_i)
        ds._convertToOneOfMany()
        # train on the current batch
        trainer.trainOnDataset(ds)

ds_all = ClassificationDataSet(93,nb_classes=9)
for index in range(len(Testdata)):
    ds_all.appendLinked(Testdata.iloc[index,:].values.tolist())
ypreds = []


for index in range(Testdata.shape[0]):
    pred = net.activate(Testdata.iloc[index,:])  
    submit.iloc[index,1:] = pred




#%% get the mean of the data
class1 = [ i for i in range(len(y)) if y[i]==1]

classOne = X.loc[class1,:].sort(axis = 0,ascending=True)


def plotMean(X,y):
    feature = {}
    featureIndex = []
    for j in range(1,10):
        classIndex = [ i for i in range(len(y)) if y[i]==j]
        print "the number of class %d is %d"  % (j, len(classIndex))
        classValue = X.loc[classIndex,:].mean()

        peakind = signal.find_peaks_cwt(classValue, np.arange(1,10))
        feature[str(j)] = peakind
        featureIndex.append(peakind)
        plt.figure()
        plt.plot(classValue)
        plt.title('The mean value of class is %d' % j)
        plt.savefig("D:/Data/Feature Selection Data OTTO/" + str(j) + '__The mean value of class.png')
    return featureIndex

        
featureIndex = plotMean(X,y)
index = set([val for lst in featureIndex for val in lst])





#%% remove outliers
from sklearn.cross_validation import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)








#%% get the weights

from sklearn.linear_model import SGDClassifier
from sklearn import cross_validation


clf = SGDClassifier(loss="log", penalty="l1",alpha =0.001)
SGD = clf.fit(X_train[:,list(index)],y_train)


from sklearn.metrics import classification_report

y_pred = SGD.predict(X_test[:,list(index)])

print(classification_report(y_test,y_pred))






#%% random forest to get the feature importance
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
#%%
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=360, max_features=1),
    AdaBoostClassifier(n_estimators=360),
    GaussianNB(),
    LDA(),
    QDA()]

RF = classifiers[5]
RF.fit(X_train[:,list(index)],y_train)
importance = RF.feature_importances_
y_pred = RF.predict(X_test[:,list(index)])

print(classification_report(y_test,y_pred))



import scipy.stats
#%% 
outliters = []
Mean = X_train.mean(axis=0)
STD = X_train.std(axis=0) 
prob = np.zeros(X_train.shape)
parameters = zip(Mean,STD)
for i in range(len(X_train[0])-1):
    colCurr = X_train[:,i]
    probability = []
    paraCurr = parameters[i]
    for item in colCurr:
        probCurr = scipy.stats.norm(paraCurr[0],paraCurr[1]).cdf(item)
        if probCurr>=0.95 or probCurr<0.05:
            probability.append(0)
        elif probCurr>0.5:
            probability.append(abs(probCurr-1)*2.0)
        else:
            probability.append(probCurr*2.0)   
    prob[:,i] = probability
# get the sum of weights
W = np.dot(prob,weights) 

    
#%%
    

def pooling(X,y):
    NewX = pd.DataFrame(columns = (X.keys()))
    label = []
    j = 0
    countNew = 0
    for i in range(1,10):
        index1 = [index for index in range(len(y)) if y[index] ==i]   
        for count in range(5,len(index1),5): 
#            print count 
            j +=1
            curr = X.iloc[countNew:count,:]
            if (len(curr.mean().dropna()) == len(curr.mean())):
                
                NewX.loc[j] = curr.mean()
            # update the left window
                countNew = count
                label.append(i)
            else:
                countNew = count
    return (NewX,label)

NewX, label = pooling(X,y)
print 'The number of training samples is %d' % NewX.shape[0]
            
    


X= NewX
y = label
#from sklearn import cross_validation
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(
#    X, y, test_size=0.4, random_state=0)

#%% Evaluation system 


#from pandas.tools.plotting import scatter_matrix
#df =  X.iloc[:,:4]
#scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')
##%% plots the data
#import pylab as plt
#
#plt.figure()
#plt.plot(X.iloc[20000,:])
#
#df.boxplot()




#plot(kind='hist', stacked=True, bins=20)

#%% traing the classifcation
from sklearn.cross_validation import train_test_split
from sklearn import svm, grid_search
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import sklearn.ensemble
RF  = sklearn.ensemble.RandomForestClassifier(random_state = 43)
#RF.fit(X,y)

#sample = RF.predict_proba(Testdata)
#%% cross validation



from sklearn.cross_validation import KFold

#skf = KFold(len(X),n_folds = 10)
#
#tuned_parameters = [{'penalty ': ['l2','l1','elasticnet'], 'gamma': [1e-3, 1e-4],
#                     'C': [1, 10, 100, 1000]},
#                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
#
#
#for train, test in kf:
#%%parameters = {'alpha':[1e-5,1e-4,1e-3,1e-2,1e-1,1e-0,1e1,1e2,1e3],
#              'penalty':['l1','elasticnet'],
#                'n_iter':[5,10,15,20,25],'loss':['log']}
#SGD = SGDClassifier()
#clf = grid_search.GridSearchCV(SGD, parameters)
#%% Random forest
parameters = {'n_estimators':[i*60 for i in range(1,5)],
             'max_features':['auto','log2',None],
'max_depth':[i*10 for i in range(1,40,8)]
}

clf = grid_search.GridSearchCV(RF, parameters)
clf = clf.fit(X,y)
clf_best = clf.best_estimator_
#clf.fit(X,y)
#GridSearchCV.fit(X,y)
#Max = 0
#for alphaValue in alpha:
#    clf = SGDClassifier(loss="log", penalty="l1",alpha =alphaValue)
#    scores = cross_validation.cross_val_score(
#    clf, X, y, cv=5)
#    accuray = scores.mean()
#    if accuray >Max:
#        Max = accuray
#        alphaFinal = alphaValue
# 
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))    
#%% predictclf_= clf.

sample = RF.predict_proba(Testdata.iloc[:,list(index)])
#%% do the DBN
#from nolearn.dbn import DBN
#clf = DBN(
#    [X.shape[1], 300, 9],
#    learn_rates=0.3,
#    learn_rate_decays=0.9,
#    epochs=10,
#    verbose=1,
#    )
#clf.fit(X,y)
#%% save to submission
submit = pd.read_csv('C:/Users/Weizhi/Desktop/Data-otto/sampleSubmission.csv')

submit.iloc[:,1:] = average
submit['id'] = submit['id'].astype(int)
submit.to_csv('C:/Users/Weizhi/Desktop/Data-otto/sample_submission_RooboostTree_SVM.csv',index = False)
