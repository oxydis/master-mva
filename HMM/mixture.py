from scipy.stats import multivariate_normal
from getdata import importdata
from kmeans import run_kmeans
import matplotlib.pyplot as plt
import numpy as np
from random import gauss
from random import random
from scipy.linalg import sqrtm
import math
from scipy import linalg
import matplotlib as mpl
from ellip import plot_cov_ellipse

"""
EM algorithm for clustering a mixture of gaussians
Author: Valentin Thomas
Date: 16/10/2015
"""

def init_param(k, iter, data):
    closest_centroids, centroids = run_kmeans(k, data, iter)
    p = np.array([1.*sum(closest_centroids == i)/closest_centroids.size for i in range(k)])
    mu = centroids
    sigma = np.array([np.cov(data[closest_centroids == i].T) for i in range(k)])
    return p, mu, sigma


def mixture(p, mu1, cov1, mu2, cov2):
    if random()<p:
        point=np.random.multivariate_normal(mu1, cov1)
    else:
        point=np.random.multivariate_normal(mu2, cov2)
    return point


def plot(p, mu, covar, data):
    colors = ['red', 'cyan', 'purple', 'blue', 'green', 'yellow']
    vec = np.array([np.array([p[i]*multivariate_normal.pdf(data[j], mu[i], sigma[i]) for j in range(len(data))]) for i in range(k)])
    clusters = np.argmax(vec, axis=0)
    data_cluster = [[] for i in range(k)]
    for j in range(k):
        for i in range(len(data)):
            if clusters[i] == j:
                data_cluster[j].append(data[i])
    splot = plt.subplot(1, 1, 1)
    for mean, covar, clust, color in zip(mu, sigma, data_cluster, colors):
        plot_cov_ellipse(covar, mean, color)
        clust = np.array(clust)
        for i in range(len(clust)):
            plt.plot(clust[i, 0], clust[i, 1], color=color, marker='o')
        plt.plot(mean[0], mean[1], color='black', marker='o')
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("EM-algorithm")
    plt.show()


def EM(k, iter, data):
    p, mu, sigma = init_param(k, iter, data)
    for i in range(iter):
        print('Call', i)
        vec = np.array([np.array([p[i]*multivariate_normal.pdf(data[j], mu[i], sigma[i]) for j in range(len(data))]) for i in range(k)])
        vec_norm = np.array([[vec[i,j]/sum(vec[:,j]) for j in range(vec.shape[1])] for i in range(vec.shape[0])])
        p = np.mean(vec_norm, axis = 1)
        p = p/sum(p)
        X=np.matrix(data)
        mu = vec_norm.dot(data)
        truc = np.sum(vec_norm, axis = 1)
        for i in range(mu.shape[0]):
            mu[i]/=truc[i]
        sigma = []
        for j in range(k):
            mat = sum([y*(x-mu[j]).T*(x-mu[j]) for y, x in zip(vec_norm[j], X)])/sum(vec_norm[j])
            sigma.append(mat)
    return p, mu, np.array(sigma)

def loglike(data, p, mu, sigma):
    vec = np.array([np.array([p[i]*multivariate_normal.pdf(data[j], mu[i], sigma[i]) for j in range(len(data))]) for i in range(k)])
    clusters = np.argmax(vec, axis=0)
    data_cluster = [[] for i in range(k)]
    for j in range(k):
        for i in range(len(data)):
            if clusters[i] == j:
                data_cluster[j].append(data[i])
    part_loglik = []
    for i in range(k):
        part_loglik.append(sum([np.log(p[i]*multivariate_normal.pdf(points, mu[i], sigma[i])) for points in data_cluster[i]]))
    return sum(part_loglik)


data, _ = importdata('EMGaussian.data')
k = 4
iter_em = 25
p, mu, sigma =  EM(k, iter_em, data)
plot(p, mu, sigma, data)
print("p", p)
print("mu", mu)
print("sigma \n", sigma)
testdata, _ = importdata('EMGaussian.test')
print('==========\n')
print('Log_lik on train data:', loglike(data, p, mu, sigma))
print('Log_lik on test data:', loglike(testdata, p, mu, sigma))
