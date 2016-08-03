

from datetime import datetime
from math import log, exp, sqrt


# TL; DR
# the main learning process start at line 122


# parameters #################################################################

train = 'C:/Users/Weizhi/github/Kaggle/click rate/train.csv'  # path to training file
test = 'C:/Users/Weizhi/github/Kaggle/click rate/test.csv'  # path to testing file


# function, generator definitions ############################################

# A. x, y generator
# INPUT:
#     path: path to train.csv or test.csv
#     label_path: (optional) path to trainLabels.csv
# YIELDS:
#     ID: id of the instance (can also acts as instance count)
#     x: a list of indices that its value is 1
#     y: (if label_path is present) label value of y1 to y33
def data(path, traindata=False):
    for t, line in enumerate(open(path)):
        # initialize our generator
        if t == 0:
            # create a static x,
            # so we don't have to construct a new x for every instance
            x = [0] * 27
            continue
        # parse x
        for m, feat in enumerate(line.rstrip().split(',')):
            if m == 0:
                ID = int(feat)
            elif traindata and m == 1:
                y = [float(feat)]
            else:
                # one-hot encode everything with hash trick
                # categorical: one-hotted
                # boolean: ONE-HOTTED
                # numerical: ONE-HOTTED!
                # note, the build in hash(), although fast is not stable,
                #       i.e., same value won't always have the same hash
                #       on different machines
                if traindata:
                    x[m] = abs(hash(str(m) + '_' + feat)) % D
                else:
                    x[m+1] = abs(hash(str(m+1) + '_' + feat)) % D

        yield (ID, x, y) if traindata else (ID, x)

# B. Bounded logloss
# INPUT:
#     p: our prediction
#     y: real answer
# OUTPUT
#     bounded logarithmic loss of p given y
def logloss(p, y):
    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)


# C. Get probability estimation on x
# INPUT:
#     x: features
#     w: weights
# OUTPUT:
#     probability of p(y = 1 | x; w)
def predict(x, w):
    wTx = 0.
    for i in x:  # do wTx
        wTx += w[i] * 1.0  # w[i] * x[i], but if i in x we got x[i] = 1.
    return 1. / (1. + exp(-max(min(wTx, 20.), -20.)))  # bounded sigmoid


# D. Update given model
# INPUT:
# alpha: learning rate
#     w: weights
#     n: sum of previous absolute gradients for a given feature
#        this is used for adaptive learning rate
#     x: feature, a list of indices
#     p: prediction of our model
#     y: answer
# MODIFIES:
#     w: weights
#     n: sum of past absolute gradients
def update(alpha, w, n, x, p, y):
    for i in x:
        # alpha / sqrt(n) is the adaptive learning rate
        # (p - y) * x[i] is the current gradient
        # note that in our case, if i in x then x[i] = 1.
        n[i] += abs(p - y)
        # (-max(min(wTx, 20.), -20.))
        w[i] = w[i]- max(min((p - y) * 1.0 * alpha / sqrt(n[i] + 1),0.9),-0.9) 

D = 2 ** 25  # number of weights use for each model, we have 32 of them
alpha = .28  # learning rate for sgd optimization
# training and testing #######################################################
start = datetime.now()

K = [0]

w = [[0.] * D]
n = [[0.] * D]

loss = 0.

tt = 1

N = 0

import random
features = random.sample(range(27),  6) 
for ID, x, y in data(train, traindata = True):
#    trainFeatures['ID'].append(ID)
#    trainFeatures['features'].append(x)
#    trainFeatures['label'].append(y)
    # get predictions and train on all labels]
#    for item in range(len(features)):
#        x += [x[item]]
    for k in K:
        p = predict(x, w[k])
        update(alpha, w[k], n[k], x, p, y[k])
        loss += logloss(p, y[k])  # for progressive validation
    
    if loss<3.75:
        N +=1
        
    if N >30 and tt *0.5 > N:
        
        
        break
    # print out progress, so that we know everything is working
    if tt % 100000 == 0:
        print('%s\tencountered: %d\tcurrent logloss: %f' % (
                datetime.now(), tt, (loss * 1./tt)))
    tt += 1
#a = 1
#import numpy as np
#np.savetext('weight.csv',w,delimiter= ",")  
#writer = csv.writer(open('trainFeatures.csv', 'wb'))
#for key, value in trainFeatures.items():
#   writer.writerow([key, value])
#trainFeatures = None
#trainFeatures.to_csv('C:/Users/Valued Customer/Desktop/data for fun/Text data/trainFeatures.csv', index = False)      
#testFeatures = {}
with open('C:/Users/Weizhi/github/Kaggle/click rate/sampleSubmission.csv', 'w') as outfile:
#    testFeatures[ID] = x
    outfile.write('id,click\n')
    for ID, x in data(test):       

#        for item in range(len(features)):
#            x += [x[item]]
        for k in K:
            p = predict(x, w[k]) + 0.00005
            outfile.write('%s,%s\n' % (ID, str(p)))
#writer = csv.writer(open('testFeatures.csv', 'wb'))
#for key, value in testFeatures.items():
#   writer.writerow([key, value])
#trainFeatures = None  
#testFeatures = None          
#testFeatures.to_csv('C:/Users/Valued Customer/Desktop/data for fun/Text data/testFeatures.csv', index = False)      
print('Done, elapsed time: %s' % str(datetime.now() - start))
