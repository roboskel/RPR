#!/usr/bin/env python

import roslib; roslib.load_manifest('attempt')
import sys
import rospy
import pointclouds
import math
import matplotlib
import scikit_dbscan
import myhog
#import myNaiveBayes
from sensor_msgs.msg import PointCloud2
from sklearn import svm


import numpy as np
import pylab as pl
pl.ion()

slot_count=0
final_data=[]
all_surf=[]
all_data=[]
all_hogs=[]
labels=[]
human_detection=[]
	
def listener():
	# parse args                                                                                                                                                               
	sub_topic = "output_cloud"

 	global mode
 	
	# create node                                                                                                                                                              
	rospy.init_node('listener')
	rospy.loginfo('Subscribing to: %s' % sub_topic)
	#rospy.loginfo('Publishing filtered clouds on: %s' % pub_topic)
	cloud_sub = rospy.Subscriber(sub_topic, PointCloud2, main_cb)

	rospy.spin()
		

def main_cb(cloud_msg):
	#DECLARE GLOBAL VARIABLES
    global slot_count

    global final_data
    global all_hogs
    global train_surfaces
    global surfacesX
    global all_surf

    global labels
	#CONVERT TO XYZ
    rospy.loginfo('converting pointcloud %d to XYZ array ',slot_count)
    raw_data = pointclouds.pointcloud2_to_xyz_array(cloud_msg, remove_nans=True)
    
    #
    mode=0  # mode=0 -->COLLECT DATA mode=1 --> TRAIN AND TEST
    #BUILD CLUSTER

    filter_zeros=np.where(raw_data[:, 0] != 0)[0]
    clear_data = raw_data[filter_zeros,:] # ignore the zeros on X
    filter_zeros=np.where(clear_data[:, 1] != 0)[0]
    clear_data = clear_data[filter_zeros,:] # ignore the zeros on Y

    cluster_labels =np.zeros((len(clear_data),1),int)
    
    #% TRAINING
    eps = 0.5
    min_points = 5
    rospy.loginfo('call DBscan ')

    [core_samples,cluster_labels, n_clusters, human, surfacesX]=scikit_dbscan.dbscan(clear_data, eps, min_points,mode,False)

    clear_data=clear_data[core_samples,:]
    # SURFACE & HOG Features EXTRACTION

    all_surf.append(surfacesX)
    rospy.loginfo('Done.')
    
    #EXTRACT AND SAVE HOGS FEATURES
    rospy.loginfo('extract hogs for timeslot %d',slot_count)
    [hogs,hog_image] = myhog.hog(surfacesX)
    #pl.plot(hog_image)
    #pl.show(0.5)
    
    final_data.append(clear_data)
    labels.append(cluster_labels)
    human_detection.append(human)   
    all_hogs.append(hogs)
    
    rospy.loginfo('all_hogs length %d',len(all_hogs))
    rospy.loginfo('final_data length %d',len(final_data))
    rospy.loginfo('human_detection length %d',len(human_detection))
    
    if mode==0:
    	f = open("train_hogs.txt","a")
    	simplejson.dump(all_hogs,f)
    	f.close() 
    	f = open("train_classifications.txt","a")
    	simplejson.dump(all_human_detection,f)
    	f.close()
    if mode==1:
    	with open("train_hogs.txt") as f:
    		train_hogs = simplejson.load(f)
    	with open("train_classifications.txt") as f2:
  	   		train_labels = simplejson.load(f2)
  	   		X=array(train_hogs)
  	   	 	y=array(train_labels)
  	   	 	clf = svm.SVC()
  	   	 	clf.fit(X, y)
  	   	 	
  	   	 	#[p0V,p1V,pAb]= trainNB0(array(train_hogs),array(train_labels))
  	   	 	#print testEntry,'classified as: ',classifyNB(array(hogs),p0V,p1V,pAb)
    slot_count = slot_count + 1
    
if __name__ == '__main__':

    listener()

