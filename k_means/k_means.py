import numpy as np 
import pandas as pd 
import random
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:
    
    def __init__(self, k=2, max_iterations=100, total_iterations=10, criterium='silhouette'):
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = None
        self.total_iterations = total_iterations
        self.criterium = criterium
        
    def fit(self, X):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        if isinstance(X, pd.DataFrame):
                X = X.to_numpy()

        best_sillhouette = 0
        best_distortion = float('inf')
        best_centroids = None

        for _ in range(self.total_iterations):
            # Initialize centroids randomly
            self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False),:]
            
            for i in range(self.max_iterations):
                # Assign each data point to the nearest centroid
                X_temp = np.expand_dims(X,axis=1)
                distances = euclidean_distance(X_temp, self.centroids)
                nearest_centroid = np.argmin(distances,axis=1)
                
                # Update centroids based on the mean of data points in each cluster
                new_centroids = np.array([X[nearest_centroid == i].mean(axis=0) for i, _ in enumerate(self.centroids)])
                
                # Stopping condition 
                if np.all(new_centroids == self.centroids):
                    break
                    
                self.centroids = new_centroids

            z = self.predict(X)

            if self.criterium == 'silhouette':
                silhouette = euclidean_silhouette(X, z)
                if silhouette > best_sillhouette:
                    best_sillhouette = silhouette
                    best_centroids = self.centroids
            elif self.criterium == 'distortion':
                distortion = euclidean_distortion(X, z)
                if distortion < best_distortion:
                    best_distortion = distortion
                    best_centroids = self.centroids
            else:
                raise Exception('not a valid criterium')

        self.centroids = best_centroids

        

        
        
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        # make X a numpy array
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        X_temp = np.expand_dims(X,axis=1) # reshaping to make broadcasting possible
        distances = euclidean_distance(X_temp, self.centroids)
        nearest_centroid = np.argmin(distances,axis=1)

        return nearest_centroid
    
    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm
        
        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """

        return self.centroids
    
    
    
    
# --- Some utility functions 

def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)

def cross_euclidean_distance(x, y=None):
    """
    
    
    """
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    distortion = 0.0
    clusters = np.unique(z)
    for i, c in enumerate(clusters):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum()
        
    return distortion


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance 
    a = D[np.arange(len(X)), z]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
    
    return np.mean((b - a) / np.maximum(a, b))

class MinMaxScaler():
    def fit(self,X):
        X = self.convert_to_DataFrame(X)
        self.min_vals = X.min()
        self.max_vals = X.max()
        self.range_vals = self.max_vals - self.min_vals

    def transform(self,X):
        X = self.convert_to_DataFrame(X)
        X.columns = self.min_vals.index # make sure both have the same column names


        return (X - self.min_vals) / self.range_vals
    
    def de_transform(self,X_transformed):
        X_transformed = self.convert_to_DataFrame(X_transformed)
        X_transformed.columns = self.min_vals.index # make sure both have the same column names
        test = X_transformed * self.range_vals + self.min_vals
        return test
    
    def convert_to_DataFrame(self,X):
        if not isinstance(X,pd.DataFrame):
            return pd.DataFrame(X)
        return X
    

        
    