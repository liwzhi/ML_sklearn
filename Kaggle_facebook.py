#

import loadingData
import pandas as pd



trainPath = 'D:/Data/facebooks data/train.csv'
testPath = 'D:/Data/facebooks data/test.csv'
bidPath = 'D:/Data/facebooks data/bids.csv'

loadingData = loadingData.loadingdata()

trainData = loadingData.loadingData(trainPath)
testData = loadingData.loadingData(testPath)
bids = loadingData.loadingData(bidPath)


#%% find the robot, data visulization
dataRobot = trainData[trainData['outcome']==1]
dataRobot0 = trainData[trainData['outcome']==0]
ID1 = dataRobot['bidder_id']
ID0 = dataRobot0['bidder_id']

behaviors = bids.groupby('bidder_id').groups



#%% data explore
featureTrain = pd.DataFrame(columns= ['country number',\
'country max','country std','country mean', 'country min'])
for i in range(5):
    DataIndex = behaviors[ID1.iloc[i]]
    print ("another ID %d" % i)
    DataSee = bids.iloc[DataIndex,:]
    dataDrop = DataSee.drop(['bid_id','time'],axis=1)
    
    for key in dataDrop:
        print 'features extraction begin'
        
        feature = DataSee.groupby(['country']).groups
        featureCount = DataSee['country'].value_counts()        
        featureTrain['country number'] = featureCount.shape[0]
        featureTrain['country max'] = featureCount.max()
        featureTrain['country std'] = featureCount.std()
        featureTrain['country min'] = featureCount.min()
        
        
        
        
        print "test country at first"
        featureKey = feature.keys()
        #%% lengthe of obejcts:
        
        
    # d6517684989560cbb0da1fb2f5bbba9b9y2st

 
    
    DataSee['country'].value_counts()
    Country = DataSee.groupby(['country']).groups
    
    
    print ('the number of country%d'%len(Country.keys()))
    
    Objects = DataSee.groupby(['merchandise']).groups
    
  
    print ("number of Objects%d" % len(Objects.keys()))
    
    #%% feature extraction
    dataDrop = DataSee.drop(['bid_id','time'],axis=1)

Counts = pd.DataFrame(index=['Object'])
Counts['Object'] = DataSee.groupby(['merchandise']).transform('count')  
#import pylab as plt
#plt.hist(DataSee['country'])
#    
#DataSee.hist(by = DataSee['country'])
#
#DataSee['merchandise'].apply(pd.value_counts).
plot(kind='bar', subplots=True)
