# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 17:52:47 2015

@author: Algorithm 001
"""

import cv2
import pylab as plt
#%% loading thed data
imgOri = cv2.imread('D:/Data/eye diagonisis/sample/sampel/4/16_left.jpeg',-1)
plt.figure()
plt.imshow(imgOri[:,:,0])


#%% pre-processing the data
Shapes = imgOri.shape
img = cv2.resize(imgOri[:,:,0],(np.int(Shapes[0]/3),np.int(Shapes[1]/3)))

imgAve = cv2.medianBlur(img,15)

plt.figure()
plt.imshow(imgAve)
plt.title('AVERAGEING IMAGE')

featureImage = {}
featureImage['rawImage'] = imgAve
#%% edge detection
from skimage.filter import sobel

from scipy import ndimage

from skimage import feature

elevation_map = sobel(imgAve)

featureImage['SobImage'] = elevation_map


plt.figure()
plt.imshow(elevation_map,cmap=plt.cm.gray,interpolation='nearest')
plt.colorbar()
plt.title('sobel IMAGE')


#plt.hist(elevation_map)
#plt.show()
##%% sobel edge detection
#elevation_map = cv2.medianBlur(elevation_map,7)
#plt.figure()
#plt.imshow(elevation_map,cmap=plt.cm.gray,interpolation='nearest')
#plt.title('edge IMAGE')
#
#
#elevation_map = feature.canny(imgAve)
#plt.figure()
#plt.imshow(elevation_map,cmap=plt.cm.gray,interpolation='nearest')
#plt.title('Canny IMAGE')

#%% hisgogarams equalization

from skimage import exposure
img_eq = exposure.equalize_hist(elevation_map)

featureImage['HE'] = elevation_map

plt.figure()
plt.imshow(img_eq,cmap=plt.cm.gray,interpolation='nearest')
plt.title('equalization IMAGE')




#elevation_map = feature.canny(img_eq)
#plt.figure()
#plt.imshow(elevation_map,cmap=plt.cm.gray,interpolation='nearest')
#plt.colorbar()
#plt.title('sobel IMAGE')

#%% fill the hole


import skimage
#imgFill = ndimage.binary_fill_holes(image)

def imshow(image,title,**kwargs):
    fig,ax = plt.subplots(figsize=(5,4))
    ax.imshow(image,**kwargs)
    ax.axis('off')
    ax.set_title(title)
imshow(img_eq, 'Original image')    
    
import numpy as np
from skimage.morphology import reconstruction
seed = np.copy(img_eq)
seed[1:-1, 1:-1] = img_eq.max()
mask = img_eq

filled = reconstruction(seed, mask, method='erosion')

imshow(filled, 'after filling holes',vmin=img_eq.min(), vmax=img_eq.max())


imshow(img_eq - filled, 'holes')
featureImage['Holes'] = img_eq - filled

# plt.title('holes')






seed = np.copy(img_eq)
seed[1:-1, 1:-1] = img_eq.min()
rec = reconstruction(seed, mask, method='dilation')
imshow(img_eq - rec, 'peaks')
plt.show()


featureImage['Peaks'] = img_eq - rec


featuresImage = featureImage.values()
#from skimage.measure import block_reduce
#featuresDown = block_reduce(featuresImages,block_size=(5,256,256),func=np.max)

featuresArray = np.zeros((featuresImage[1].shape[0],featuresImage[1].shape[1],5))

for i in range(5):
    featuresArray[:,:,i] = featuresImage[i]
featuresArray = featuresArray[::2,::2,:]    
X = np.reshape(featuresArray,(featuresArray.shape[0]*featuresArray.shape[1],5))

imshow(featuresArray[:,:,4],'image')
plt.figure()
plt.imshow(featuresArray[:,:,4],cmap = plt.cm.gray)
#%% image semenation 

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_norm = min_max_scaler.fit_transform(X)
X = X_norm
#Keys = featureImage.keys()
#for key in Keys:
#    featureImage['HE'] = 

print(__doc__)

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs

###############################################################################
# Generate sample data



###############################################################################
# Compute clustering with MeanShift

# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)



labels = ms.labels_

for i in set(np.unique(labels)):
    Index = np.where(labels==i)
    print 'the numerb of %d is %d' % (i,Index[0].shape[0])
    for index in Index:
        labels[index] = (0.1+i*2)*10



psudoImage = np.reshape(labels,(featuresArray.shape[0],featuresArray.shape[1]))

plt.imshow(psudoImage,cmap=plt.cm.jet)
plt.colorbar()

imshow(psudoImage,'means')



cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

###############################################################################
# Plot result
import matplotlib.pyplot as plt
from itertools import cycle

plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

#%% feature extraction from these images


#%% do the aveageing again
#from skimage.morphology import disk
#from skimage.filters import rank
#selem = disk(20)
#bilateral_result = rank.mean_bilateral(img_eq, selem=selem, s0=500, s1=500)
#
#plt.figure()
#plt.imshow(np.hstack((img_eq, bilateral_result)))
#plt.title('Bilateral mean')
#plt.axis('off')
#
##%% texture detection
#def power(image, kernel):
#    # Normalize images for better comparison.
#    image = (image - image.mean()) / image.std()
#    return np.sqrt(nd.convolve(image, np.real(kernel), mode='wrap')**2 +
#                   nd.convolve(image, np.imag(kernel), mode='wrap')**2)
#image = cv2.resize(img_eq - filled,(256,256))
#kernel_params = []
#results1 = []
#from skimage.filters import gabor_kernel
#for theta in (set(np.arange(0,8,0.5))):
#    theta = theta / 4. * np.pi
#    for frequency in (0.1, 0.2):
#        kernel = gabor_kernel(frequency, theta=theta)
#        params = 'theta=%d,\nfrequency=%.2f' % (theta * 180 / np.pi, frequency)
#        kernel_params.append(params)
#        # Save kernel and the power image for each image
#        results1.append(power(image, kernel))
#
#
##%% output        
#imgResult = results1[:16]
#imgMax = np.zeros((256,256,16))
#for i in range(16):
#    imgMax[:,:,i] = imgResult[i]
#imgmax = np.reshape(imgMax,(256*256,16))
#imgmax = np.reshape(np.amax(imgmax,axis=1),(256,256))
#
#
#plt.figure;plt.imshow(results1[1],cmap=plt.cm.gray, interpolation='nearest')
#
#
#
#
##%% segmenation algorithm 
#
#
#
##%% update whole features
#
#
#
##%% classifcation
#
#
##%% see the histgrome
#
#hist = np.histogram(img,bins = np.arange(0,256))
#fig, (ax1,ax2) = plt.subplots(1,2,figsize=(8,3))
#
#ax1.imshow(img,cmap=plt.cm.gray,interpolation='nearest')
#ax1.axis('off')
#
#ax2.plot(hist[1][:-1],hist[0],lw=2)
#ax2.set_title("histogram of grey values")
#
#
##%% find the local min -- marker
#from skimage.feature import peak_local_max
#
#
#distance = ndimage.distance_transform_edt(img)
#local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
#                            labels=img)
#
#markers = ndimage.label(local_maxi)[0]
#
#labels = watershed(-distance, markers, mask=img)
#plt.figure()
#plt.imshow(labels,cmap=plt.cm.gray,interpolation='nearest')
#
##%% fill the holds
#from scipy import ndimage
#fill_coins = ndimage.binary_fill_holes(labels)
#
#fig,ax = plt.subplots(figsize=(4,3))
#ax.imshow(fill_coins,cmap=plt.cm.gray,interpolation='nearest')
#ax.axis('off')
#ax.set_title('canny detect with filling the holes')
#
##%% using the canny
#from skimage.filter import canny
#
#img = img[:,:,1]
#edges = canny(img/255.)
#
#fig,ax = plt.subplots(figsize=(4,3))
#ax.imshow(edges,cmap=plt.cm.gray,interpolation='nearest')
#ax.axis('off')
#ax.set_title('canny detect')
##%% fill the holds
#from scipy import ndimage
#fill_coins = ndimage.binary_fill_holes(edges)
#
#fig,ax = plt.subplots(figsize=(4,3))
#ax.imshow(fill_coins,cmap=plt.cm.gray,interpolation='nearest')
#ax.axis('off')
#ax.set_title('canny detect with filling the holes')
#
##%% remove object that smaller than threshold
#label_objects,np_labels = ndimage.label(fill_coins)
#
#size = np.bincount(label_objects.ravel())
#
#mask_sizes = size>20
#
#mask_sizes[0] = 0
#
#coins_cleared = mask_sizes[label_objects]
#
#fig,ax = plt.subplots(figsize=(4,3))
#ax.imshow(coins_cleared,cmap=plt.cm.gray,interpolation='nearest')
#ax.axis('off')
#ax.set_title('canny detect with filling the holes')






