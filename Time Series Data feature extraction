import pandas as pd
import numpy as np
class featuerExtraction:
    def featureExtract(self,data):
        maxIndex = np.argmax(data)
#        maxIndex = data.idxmax()
        feature = {}
        feature['max'] = data.max()
        feature['mean'] = data.mean()
        feature['std']  = data.std()
        feature['maxOverMean'] = data.max()/float(data.mean())
        feature['stdOverMean'] = data.std()/float(data.mean())   
        # for mean to get the number of crossing:
        crossing = self.numberCrossing(data,0)
        feature['zeroCrossing_0'] = len(crossing)
        feature['lengthOfCrossing_0'] = self.lengthCrossing(crossing,maxIndex)
        power,powerPercent= self.sumOfPower(data,crossing,maxIndex)
        feature['power_0'] = power
        feature['powerPercent_0'] = powerPercent
        # for mean + std to get the number of crossing:
        crossing = self.numberCrossing(data,data.std())
        feature['zeroCrossing_1'] = len(crossing)
        feature['lengthOfCrossing_1'] = self.lengthCrossing(crossing,maxIndex)   
        power,powerPercent= self.sumOfPower(data,crossing,maxIndex)
        feature['power_1'] = power
        feature['powerPercent_1'] = powerPercent
        # for mean + 2*std to get the number of crossing:
        crossing = self.numberCrossing(data,2*data.std())
        feature['zeroCrossing_2'] = len(crossing)
        feature['lengthOfCrossing_2'] = self.lengthCrossing(crossing,maxIndex)  
        power,powerPercent= self.sumOfPower(data,crossing,maxIndex)
        feature['power_2'] = power
        feature['powerPercent_2'] = powerPercent
        return pd.DataFrame(np.array(feature.values()).T,index=feature.keys())
#        return feature
    def numberCrossing(self,data,value):
        output = []
        for i in range(len(data)-1):
            valueMean = data.mean() + value
            left = (data[i]-valueMean)
            right = (data[i+1] -valueMean)
            if (left<0 and right>0) or (left>0 and right<0) or left==0:
                output.append(i)   
        return output
    def lengthCrossing(self,crossing,maxIndex):
        Value = 0
        for j in range(len(crossing)-1):
            left = (crossing[j]-maxIndex)
            right = (crossing[j+1]-maxIndex)   
            if left<0 and right>0:
                Value = abs(crossing[j]-crossing[j+1])
        return Value
    def sumOfPower(self,data,crossing,maxIndex):
        power = 0
        powerSum = 0
        Value = [65,128]
        for j in range(len(crossing)-1):
            left = (crossing[j]-maxIndex)
            right = (crossing[j+1]-maxIndex)  
            if left<0 and right>0:
                Value = [crossing[j],crossing[j+1]]
#                print Value
                break
        for i in range(Value[0],Value[1]):
            power += data[i]**(2)   
        for i in range(len(data)):
            powerSum += data[i]**(2)
        return (power,power/float(powerSum))        
#a = featuerExtraction()
#output = pd.DataFrame()
#output = a.featureExtract(data)
