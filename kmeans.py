#%%
# K-means clustering algo from scratch
#

import numpy as np
import matplotlib.pyplot as plt
import scipy

class KMeans:
    
    def __init__(self, dataset, k, iterations=100):
        #number of iterations to run algo
        self.iterations = iterations
        #number of clusters
        self.k = k 
        
        # returns i points that are centers of clusters
        self.centroids = dataset[np.random.choice(dataset.shape[0], size=k, replace=False)]
        
        # contains list of length i that has index to a cluster for every point in the dataset. Will be initialized
        self.clusters = {x:[] for x in range(self.k)}
                
        while self.iterations > 0:
            self.__fit_points(dataset)
            self.iterations -= 1
        
    # one iteration of clustering
    def __fit_points(self, points):
        # calculate distance of every point to centroids
        distances = scipy.spatial.distance.cdist(points, self.centroids)
        
        # set index of closest centroid for every point
        self.__create_clusters(points, distances)
        
        # calculate new centroid by taking average of every point in each cluster
        self.__set_centroids()
        
    #calculates average between multiple points
    def __set_centroids(self):
        for c in range(self.k):
            self.centroids[c] = np.mean(np.array(self.clusters[c]))
    
    def __create_clusters(self, points, distances):
        closest_cluster = np.argmin(distances, axis=1)
        for c in range(self.k):
            self.clusters[c] = points[np.where(c == closest_cluster)]
        
    # for all points in x
    # returns the class/cluster it is closest to
    #
    # plot also plots the points x into the clusters
    def predict(self, x, plot=False):
        if plot:
            plt.scatter(self.x[:,0], self.x[:,-1], color="red", alpha=0.5)
            
        distance_to_clusters = scipy.spatial.distance.cdist(x, self.centroids)
        closest_cluster = np.argmin(distance_to_clusters, axis=1) #returned from function
        return closest_cluster
        
    # plots the clusters
    def plot(self):
        colors = np.random.rand(self.k,3)
        
        plt.scatter(self.centroids[:,0], self.centroids[:,-1], color="black")

        for k in self.clusters:
            plt.scatter(self.clusters[k][:, 0], self.clusters[k][:, -1], color=tuple(colors[k]), alpha=0.5)
        plt.show()
    

# Testing out
N = 500
K = 4
points = np.random.rand(300, 2)

model = KMeans(points, K, N)
model.plot()

