# -*- coding: utf-8 -*-
"""
DBSCAN: Density-Based Spatial Clustering of Applications with Noise
"""

# Author: Robert Layton <robertlayton@gmail.com>
#
# License: BSD 3 clause
import rospy
import sys
import my_griddata
import numpy as np
import pylab as pl

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_random_state
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D


pl.ion()
def dbscan(X, eps, min_samples, mode, visualize,metric='minkowski',
           algorithm='auto', leaf_size=30, p=2, random_state=None):
    """Perform DBSCAN clustering from vector array or distance matrix.

Parameters
----------
X: array [n_samples, n_samples] or [n_samples, n_features]
Array of distances between samples, or a feature array.
The array is treated as a feature array unless the metric is given as
'precomputed'.

eps: float, optional
The maximum distance between two samples for them to be considered
as in the same neighborhood.

min_samples: int, optional
The number of samples in a neighborhood for a point to be considered
as a core point.

metric: string, or callable
The metric to use when calculating distance between instances in a
feature array. If metric is a string or callable, it must be one of
the options allowed by metrics.pairwise.calculate_distance for its
metric parameter.
If metric is "precomputed", X is assumed to be a distance matrix and
must be square.

algorithm: {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
The algorithm to be used by the NearestNeighbors module
to compute pointwise distances and find nearest neighbors.
See NearestNeighbors module documentation for details.

leaf_size: int, optional (default = 30)
Leaf size passed to BallTree or cKDTree. This can affect the speed
of the construction and query, as well as the memory required
to store the tree. The optimal value depends
on the nature of the problem.

p: float, optional
The power of the Minkowski metric to be used to calculate distance
between points.

random_state: numpy.RandomState, optional
The generator used to initialize the centers. Defaults to numpy.random.

Returns
-------
core_samples: array [n_core_samples]
Indices of core samples.

labels : array [n_samples]
Cluster labels for each point. Noisy samples are given the label -1.

Notes
-----
See examples/cluster/plot_dbscan.py for an example.

References
----------
Ester, M., H. P. Kriegel, J. Sander, and X. Xu, “A Density-Based
Algorithm for Discovering Clusters in Large Spatial Databases with Noise”.
In: Proceedings of the 2nd International Conference on Knowledge Discovery
and Data Mining, Portland, OR, AAAI Press, pp. 226–231. 1996
"""
    if not eps > 0.0:
        raise ValueError("eps must be positive.")

    X = np.asarray(X)
    
    n = X.shape[0]

    # If index order not given, create random order.
    random_state = check_random_state(random_state)
    index_order = np.arange(n)
    random_state.shuffle(index_order)

    # check for known metric powers
    distance_matrix = True
    if metric == 'precomputed':
        D = pairwise_distances(X, metric=metric)
    else:
        distance_matrix = False
        neighbors_model = NearestNeighbors(radius=eps, algorithm=algorithm,
                                           leaf_size=leaf_size,
                                           metric=metric, p=p)
        neighbors_model.fit(X)

    # Calculate neighborhood for all samples. This leaves the original point
    # in, which needs to be considered later (i.e. point i is the
    # neighborhood of point i. While True, its useless information)
    neighborhoods = []
    if distance_matrix:
        neighborhoods = [np.where(x <= eps)[0] for x in D]

    # Initially, all samples are noise.
    labels = -np.ones(n)

    # A list of all core samples found.
    core_samples = []

    # label_num is the label given to the new cluster
    label_num = 0

    # Look at all samples and determine if they are core.
    # If they are then build a new cluster from them.
    for index in index_order:
        # Already classified
        if labels[index] != -1:
            continue

        # get neighbors from neighborhoods or ballTree
        index_neighborhood = []
        if distance_matrix:
            index_neighborhood = neighborhoods[index]
        else:
            index_neighborhood = neighbors_model.radius_neighbors(
                X[index], eps, return_distance=False)[0]

        # Too few samples to be core
        if len(index_neighborhood) < min_samples:
            continue

        core_samples.append(index)
        labels[index] = label_num
        # candidates for new core samples in the cluster.
        candidates = [index]

        while len(candidates) > 0:
            new_candidates = []
            # A candidate is a core point in the current cluster that has
            # not yet been used to expand the current cluster.
            for c in candidates:
                c_neighborhood = []
                if distance_matrix:
                    c_neighborhood = neighborhoods[c]
                else:
                    c_neighborhood = neighbors_model.radius_neighbors(
                        X[c], eps, return_distance=False)[0]
                noise = np.where(labels[c_neighborhood] == -1)[0]
                noise = c_neighborhood[noise]
                labels[noise] = label_num
                for neighbor in noise:
                    n_neighborhood = []
                    if distance_matrix:
                        n_neighborhood = neighborhoods[neighbor]
                    else:
                        n_neighborhood = neighbors_model.radius_neighbors(
                            X[neighbor], eps, return_distance=False)[0]
                    # check if its a core point as well
                    if len(n_neighborhood) >= min_samples:
                        # is new core point
                        new_candidates.append(neighbor)
                        core_samples.append(neighbor)
            # Update candidates for next round of cluster expansion.
            candidates = new_candidates
        # Current cluster finished.
        # Next core point found will start a new cluster.
        label_num += 1

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d' % n_clusters_)
   # print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))

##############################################################################
# Plot result
# Black removed and is used for noise instead.
    human = np.zeros((1, len(labels)),int)    #declare all points 0
    unique_labels = set(labels)
    surface=[]
    #colors = pl.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    #ccnames=['grey', 'black', 'violet', 'blue', 'cyan', 'rosy', 'orange', 'red', 'green', 'brown', 'yellow', 'gold']
    ccnames =['blue','green','red','cyan','magenta','yellow','black','white','grey']
    cc = ['b','g','r','c','m','y','k','w','0.75' ]
    
    [xi,yi,zi] = [X[:,0] , X[:,1] , X[:,2]]
    
    [xmin, xmax] = [min(xi), max(xi)]
    [ymin, ymax] = [min(yi), max(yi)]
    [zmin, zmax] = [min(zi), max(zi)]
    [xnodes, ynodes, znodes] = [np.linspace(xmin, xmax, 20, endpoint=True), np.linspace(ymin, ymax, 20, endpoint=True), np.linspace(zmin, zmax, 20, endpoint=True)]
    #cc = [190 190 190, 0 0 0, 138 43 226, 0 0 255, 0 255 255, 255 193 193,   255 127 0, 255 0 0,0 255 0, 139 69 19, 255 255 0, 139    117    0]./255;
    for k, col in zip(unique_labels, cc):
        if k == -1:
            # Black used for noise.
            col = 'k'
            markersize = 6
        class_members = [index[0] for index in np.argwhere(labels == k)]
        
        cluster_core_samples = [index for index in core_samples
                                if labels[index] == k]
        
        for index in class_members:
            x = X[index]
            if index in core_samples and k != -1:
                markersize = 10
            else:
                markersize = 6  
             
            
            pl.plot(x[1], x[2], 'o', markerfacecolor=col,markeredgecolor='k', markersize=markersize)
            
    pl.title('Estimated number of clusters: %d' % n_clusters_)
    pl.show() #plot figure
    #MANUALLY ANNOTATE DATA
    
    obj = 0
    for obj in range(0,n_clusters_):
        filter=np.where(labels[:]==obj)[0]

        if mode==0:
            rospy.loginfo('Is %s human (1 for yes, 0 for no): ', ccnames[obj])
            temp=input()
            for i in filter:
                human[0,i] = temp   

        surface= my_griddata.griddata(yi[filter], zi[filter], xi[filter], ynodes, znodes,'nn')
        
        #surface=surface-min(min(surface))
        rospy.loginfo('extract surface for cluster %d', obj)
             
        obj = obj + 1
    pl.close()
    return core_samples, labels, n_clusters_, human, surface


class DBSCAN(BaseEstimator, ClusterMixin):
    """Perform DBSCAN clustering from vector array or distance matrix.

DBSCAN - Density-Based Spatial Clustering of Applications with Noise.
Finds core samples of high density and expands clusters from them.
Good for data which contains clusters of similar density.

Parameters
----------
eps : float, optional
The maximum distance between two samples for them to be considered
as in the same neighborhood.
min_samples : int, optional
The number of samples in a neighborhood for a point to be considered
as a core point.
metric : string, or callable
The metric to use when calculating distance between instances in a
feature array. If metric is a string or callable, it must be one of
the options allowed by metrics.pairwise.calculate_distance for its
metric parameter.
If metric is "precomputed", X is assumed to be a distance matrix and
must be square.
random_state : numpy.RandomState, optional
The generator used to initialize the centers. Defaults to numpy.random.

Attributes
----------
`core_sample_indices_` : array, shape = [n_core_samples]
Indices of core samples.

`components_` : array, shape = [n_core_samples, n_features]
Copy of each core sample found by training.

`labels_` : array, shape = [n_samples]
Cluster labels for each point in the dataset given to fit().
Noisy samples are given the label -1.

Notes
-----
See examples/plot_dbscan.py for an example.

References
----------
Ester, M., H. P. Kriegel, J. Sander, and X. Xu, “A Density-Based
Algorithm for Discovering Clusters in Large Spatial Databases with Noise”.
In: Proceedings of the 2nd International Conference on Knowledge Discovery
and Data Mining, Portland, OR, AAAI Press, pp. 226–231. 1996
"""

    def __init__(self, eps, min_samples, metric='euclidean',
                 algorithm='auto', leaf_size=30, p=None, random_state=None):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.random_state = random_state

    def fit(self, X):
        """Perform DBSCAN clustering from features or distance matrix.

Parameters
----------
X: array [n_samples, n_samples] or [n_samples, n_features]
Array of distances between samples, or a feature array.
The array is treated as a feature array unless the metric is
given as 'precomputed'.
params: dict
Overwrite keywords from __init__.
"""
        clust = dbscan(X, **self.get_params())
        self.core_sample_indices_, self.labels_ = clust
        self.components_ = X[self.core_sample_indices_].copy()
        return self
