# -*- coding: utf-8 -*-
"""
Created on Sat May 16 15:29:12 2015

@author: craig
"""


import sorting_with_python as swp
import numpy as np
import matplotlib.pylab as plt


# Create a list with the file names
data_files_names = ['Locust_' + str(i) + '.dat' for i in range(1,5)]


# Get the length of the data in the files
data_len = np.unique(list(map(len, map(lambda n: np.fromfile(n,np.double), data_files_names))))[0]


# Load the data in a list of numpy arrays
data = [np.fromfile(n,np.double) for n in data_files_names]


# Check quantiles of recordings
from scipy.stats.mstats import mquantiles
np.set_printoptions(precision=3)
[mquantiles(x, prob=[0,0.25,0.5,0.75,1]) for x in data]


# Check standard deviation
[np.std(x) for x in data]


# What is the amplitude step size
[np.min(np.diff(np.sort(np.unique(x)))) for x in data]


# Plot a section of the data
fs = 1.5e4
tt = np.arange(0,data_len)/fs
swp.plot_data_list(data,tt,0.1)
plt.xlim([0, 0.2])


# Data renormalization
data_mad = list(map(swp.mad, data))
data_mad


# Apply MAD data normalization
data = list(map(lambda x: (x-np.median(x))/swp.mad(x), data))


# Plot and compare MAD and SD normalization
plt.plot(tt, data[0], color = "black")
plt.xlim([0, 0.2])
plt.ylim([-17, 13])
plt.axhline(y=1, color="red")
plt.axhline(y=-1, color="red")
plt.axhline(y=np.std(data[0]), color="blue", linestyle="dashed")
plt.axhline(y=-np.std(data[0]), color= "blue", linestyle="dashed")
plt.xlabel('Time (s)')
plt.ylim([-5, 10])


# Check the MAD procedure
dataQ = map(lambda x: mquantiles(x, prob=np.arange(0.01, 0.99, 0.001)), data)
dataQsd = map(lambda x: mquantiles(x/np.std(x), prob=np.arange(0.01, 0.99, 0.001)), data)

from scipy.stats import norm
qq = norm.ppf(np.arange(0.01, 0.99, 0.001))
plt.plot(np.linspace(-3, 3, num=100), np.linspace(-3,3,num=100), color= 'grey')

colors = ['black', 'orange', 'blue', 'red']

for i,y in enumerate(dataQ):
    plt.plot(qq,y,color=colors[i])
    
for i,y in enumerate(dataQsd):
    plt.plot(qq,y,color=colors[i], linestyle='dashed')
    
plt.xlabel('Normal quantiles')
plt.ylabel('Empirical quantiles')



# Detect peaks
from scipy.signal import fftconvolve
from numpy import apply_along_axis as apply


# Smooth and threshold data
data_filtered = apply(lambda x: fftconvolve(x, np.array([1,1,1,1,1])/5.0,'same'), 1, np.array(data))
data_filtered = (data_filtered.transpose() / apply(swp.mad,1,data_filtered)).transpose()
data_filtered[data_filtered < 4] = 0


plt.plot(tt, data[0], color='black')
plt.axhline(y=4, color='blue', linestyle = 'dashed')
plt.plot(tt, data_filtered[0,], color='red')
plt.xlim([0, 0.2])
plt.ylim([-5, 10])
plt.xlabel('Time (s)')


# Get peaks
spikes0 = swp.peak(data_filtered.sum(0))


# Interactive spike detection check
swp.plot_data_list_and_detection(data,tt,spikes0)
plt.xlim([0,0.2])


# Split data into two parts
spikes1 = spikes0[spikes0 <= data_len/2.0]
spikes2 = spikes0[spikes0 > data_len/2.0]





# Cuts
evtsE = swp.mk_events(spikes1, np.array(data),49,50)
evtsE_median = apply(np.median,0,evtsE)
evtsE_mad = apply(swp.mad, 0, evtsE)


plt.plot(evtsE_median, color='red', lw=2)
plt.axhline(y=0, color= 'black')
for i in np.arange(0, 400, 100):
    plt.axvline(x=i, color='black', lw=2)
    
for i in np.arange(0,400,10):
    plt.axvline(x=i,color='grey')
    
plt.plot(evtsE_median, color = 'red', lw=2)
plt.plot(evtsE_mad, color='blue', lw=2)




# Get waveform segments centered about spike time
evtsE = swp.mk_events(spikes0, np.array(data),14,30)

# Examine 200 waveforms
swp.plot_events(evtsE,200)


# Examine noise "events" between spike events
noiseE = swp.mk_noise(spikes0, np.array(data), 14, 30, safety_factor=2.5, size=2000)


# Function to get events that aren't overlapping. These events occur
# in isolation and are "clean"
def good_evts_fct(samp,thr=3):
    samp_med = apply(np.median,0,samp)
    samp_mad = apply(swp.mad,0,samp)
    above = samp_med > 0
    samp_r = samp.copy()
    
    for i in range(samp.shape[0]): samp_r[i,above] = 0
        
    samp_med[above] = 0
    res = apply(lambda x: np.all(abs((x-samp_med)/samp_mad)<thr), 1, samp_r)
    return(res)


# Get clean events, and plot
goodEvts = good_evts_fct(evtsE,8)
temp = evtsE[goodEvts,:]
swp.plot_events(temp[:200,:])


# Check "bad" events
badtemp = evtsE[-goodEvts,:]
swp.plot_events(badtemp, show_median=False, show_mad=False)




# Perform dimension reduction using PCA to get spike models
from numpy.linalg import svd
varcovmat = np.cov(evtsE[goodEvts, :].T)

u, s, v = svd(varcovmat)



evt_idx = range(180)
evtsE_good_mean = np.mean(evtsE[goodEvts,:],0)

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(evt_idx, evtsE_good_mean, 'black', 
             evt_idx, evtsE_good_mean + 5 * u[:,i], 'red',
             evt_idx, evtsE_good_mean - 5 * u[:,i], 'blue')
    #plt.title('PC' + str(i) + ': ' str(round(s[i]/sum(s)*100))+'%')
    
    # What's the syntax for the title string in Python 2?
    

for i in range(4,8):
    plt.subplot(2,2,i-3)
    plt.plot(evt_idx, evtsE_good_mean, 'black', 
             evt_idx, evtsE_good_mean + 5 * u[:,i], 'red',
             evt_idx, evtsE_good_mean - 5 * u[:,i], 'blue')
            
#    plt.title()
    

noiseVar = sum(np.diag(np.cov(noiseE.T)))    
evtsVar = sum(s)
[(i,sum(s[:i])+noiseVar-evtsVar) for i in range(15)]



# Static representation of projected data
evtsE_good_P0_to_P3 = np.dot(evtsE[goodEvts, :], u[:,0:4])
from pandas.tools.plotting import scatter_matrix
import pandas as pd
df = pd.DataFrame(evtsE_good_P0_to_P3)
scatter_matrix(df, alpha=0.2, s=4, c='k', figsize=(6,6), 
               diagonal='kde', marker=".")


# Clustering with K-Means
from sklearn.cluster import KMeans
km10 = KMeans(n_clusters=10, init = 'k-means++', n_init=100, max_iter=100)
km10.fit(np.dot(evtsE[goodEvts, :], u[:,0:3]))
c10 = km10.fit_predict(np.dot(evtsE[goodEvts,:],u[:,0:3]))

# Sort by absolute value of the median
cluster_median = list([(i, np.apply_along_axis(np.median,0,
            evtsE[goodEvts,:][c10 == i,:]))
            for i in range(10)
            if sum(c10 == i) > 0])



cluster_size = list([np.sum(np.abs(x[1])) for x in cluster_median])
# Code has a bug in next two lines, so ignore
#new_order = list(reversed(np.argsort(cluster_size)))
#new_order_reverse = sorted(range(len(new_order)), key=new_order._getitem_)
#c10b = [new_order_reverse[i] for i in c10]
c10b = c10


# Cluster specific plots - first 5 clusters
plt.subplot(511)
swp.plot_events(evtsE[goodEvts,:][np.array(c10b) == 0,:])
plt.ylim([-15,20])
plt.subplot(512)
swp.plot_events(evtsE[goodEvts,:][np.array(c10b) == 1,:])
plt.ylim([-15,20])
plt.subplot(513)
swp.plot_events(evtsE[goodEvts,:][np.array(c10b) == 2,:])
plt.ylim([-15,20])
plt.subplot(514)
swp.plot_events(evtsE[goodEvts,:][np.array(c10b) == 3,:])
plt.ylim([-15,20])
plt.subplot(515)
swp.plot_events(evtsE[goodEvts,:][np.array(c10b) == 4,:])
plt.ylim([-15,20])


# Cluster specific plots - last 5 clusters
plt.subplot(511)
swp.plot_events(evtsE[goodEvts,:][np.array(c10b) == 5,:])
plt.ylim([-10,10])
plt.subplot(512)
swp.plot_events(evtsE[goodEvts,:][np.array(c10b) == 6,:])
plt.ylim([-10,10])
plt.subplot(513)
swp.plot_events(evtsE[goodEvts,:][np.array(c10b) == 7,:])
plt.ylim([-10,10])
plt.subplot(514)
swp.plot_events(evtsE[goodEvts,:][np.array(c10b) == 8,:])
plt.ylim([-10,10])
plt.subplot(515)
swp.plot_events(evtsE[goodEvts,:][np.array(c10b) == 9,:])
plt.ylim([-10,10])



# code below appears to have bugs - stop here

# Spike peeling: brute force superposition resolution
centers = {"Cluster " + str(i): swp.mk_center_dictionary(spikes1[goodEvts][np.array(c10b)==i], np.array(data)) for i in range(10)}



# First peeling
swp.classify_and_align_evt(spikes0[0], np.array(data), centers)

data0 = np.array(data)
round0 = [swp.classify_and_align_evt(spikes0[i],data0,centers) for i in range(len(spikes0))]

















