from getdata import importdata
import matplotlib.pyplot as plt
import numpy as np

"""
K-Means algorithm
Author: Valentin Thomas
Date: 23/10/2015

Computes K-means on 2D variables.
"""

def distance(point, centroids, data):
    return np.array([np.linalg.norm(point-center) for center in centroids])


def init_centroids(k, data):
    index = np.random.randint(data.shape[0], size = k)
    return np.array([data[i] for i in index])


def compute_centroids(closest_centroids, k, data):
    classes = []
    for i in range(k):
        tmp = np.array([data[j] for j in range(closest_centroids.size) if closest_centroids[j] == i])
        classes.append(tmp)
    return np.array([np.mean(e, axis=0) for e in classes])


def find_closest_centroid(centroids, data):
    closest_centroids = np.array([np.argmin(distance(point, centroids, data)) for point in data])
    return closest_centroids


def run_kmeans(k, data, iter):
    centroids = init_centroids(k, data)
    for i in range(iter):
        closest_centroids = find_closest_centroid(centroids, data)
        centroids = compute_centroids(closest_centroids, k, data)
        print("K-means iter=", i+1, "/", iter)
        print("Distorsion = ", distorsion(closest_centroids, centroids, k, data))
    return closest_centroids, centroids


def plot_clusters(closest_centroids, centroids, k, data, lines=False, title='Clustering with k-means'):
    marker_data = '-o' if lines else 'o'
    for i in range(k):
        tmp = np.array([data[j] for j in range(closest_centroids.size) if closest_centroids[j] == i])
        plt.plot(tmp[:,0], tmp[:,1], marker_data)
    for i in range(k):
        plt.plot(centroids[i,0], centroids[i,1], color = 'black', marker='o')
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title(title)
    plt.show()
    return

def distorsion(closest_centroids, centroids, k, data):
    dist = 0
    for i in range(k):
        for n in range(len(data)):
            if closest_centroids[n] == i:
                dist += np.linalg.norm(data[n]-centroids[i])**2
    return dist/len(data)


if __name__ == '__main__':
    #np.random.seed(0)
    data, _ = importdata('EMGaussian.data')
    k = 4
    closest_centroids, centroids = run_kmeans(k, data, 10)
    print("Distorsion = ", distorsion(closest_centroids, centroids, k, data))
    plot_clusters(closest_centroids, centroids, k, data)

