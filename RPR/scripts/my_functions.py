#!/usr/bin/env python

import roslib; roslib.load_manifest('attempt')
import sys
import rospy
import pointclouds
import mydbscan
import math
import matplotlib
import scikit_dbscan
import myhog

from sensor_msgs.msg import PointCloud2
import my_griddata

import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ccnames =['blue','green','red','cyan','magenta','yellow','black']
cc = ['b','g','r','c','m','y','k']

def do_cluster(raw_data, mode):
    
    cluster_labels=[]
    filter_zeros=np.where(raw_data[:, 1] != 0)[0]
    #% OPTIMIZE DATA QUALITY
    clear_data = raw_data[filter_zeros,:] # ignore the zeros on X
    filter_zeros=np.where(clear_data[:, 2] != 0)[0]
    clear_data = clear_data[filter_zeros,:] # ignore the zeros on Y
    rospy.loginfo('filtered data')
    #filter2=np.where(clear_data[:, 1] > 0.5)[0] 
    #clear_data = clear_data[filter2, :]    # ignore mishits 
    
    #% TRAINING 
    if mode == 0:
 
        cluster_labels =np.zeros((len(clear_data),1),int)
        eps = 0.5
        min_points = 3
        
        rospy.loginfo('call DBscan ')
        [core_samples,cluster_labels, n_clusters, human]=scikit_dbscan.dbscan(clear_data, eps, min_points,mode) 


    #% TESTING
    if mode == 1:
        
        rospy.loginfo('call DBscan ')
        [core_samples,cluster_labels,n_clusters, human]=scikit_dbscan.dbscan(clear_data,eps, min_points)
        

    return core_samples,cluster_labels,human

#def extract_surface_x(cluster_labels, all):
#    [x,y,z] = [all[:,0],all[:,1],all[:,2]]
#    
#    [xmin, xmax] = [min(x), max(x)]
#    [ymin, ymax] = [min(y), max(y)]
#    [xnodes, ynodes] = [np.linspace(xmin, xmax, 10, endpoint=True), np.linspace(ymin, ymax, 10, endpoint=True)]
#   # Surface fitting.
#    khat = max(cluster_labels)
#   g = [] #create object list
#   cnt = 1
 #  
   # for i in mslice[1:khat]:
  #      if numel(find(cluster_labels == i)) > 40:
   #         a=all[cluster_labels == i, 2]
      #      b=all[cluster_labels == i, 3]
     #       c=all[cluster_labels == i, 1]
    #        g[1]= my_griddata.griddata(a, b, c, xnodes, ynodes,'nn')
            #g[1]=g[1]-min(g[1])
       #     cnt = cnt + 1 
    #return g
