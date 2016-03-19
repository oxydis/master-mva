"""
Author: Valentin Thomas
Script for the QDA maximum likelihood estimation
"""

import numpy as np
import matplotlib.pyplot as plt
from getdata import importdata


def train_model(trainX, trainY):
	trainX0=np.array([x for x,y in zip(trainX,trainY) if y==0])
	trainX1=np.array([x for x,y in zip(trainX,trainY) if y==1])

	### QDA
	piML=1.*trainX1.shape[0]/trainX.shape[0]
	mu0ML=np.mean(trainX0, axis=0)
	mu1ML=np.mean(trainX1, axis=0)
	sigma0ML=(sum([np.outer(x0-mu0ML, x0-mu0ML)for x0 in trainX0]))/trainX0.shape[0]
	sigma1ML = (sum([np.outer(x1-mu1ML, x1-mu1ML)for x1 in trainX1]))/trainX1.shape[0]
	print "PiML:", piML, "\nmu0ML:", mu0ML, "\nmu1ML:", mu1ML, "\nSigma0ML:\n", sigma0ML, "\nSigma1ML:\n", sigma1ML
	return piML, mu0ML, mu1ML, sigma0ML, sigma1ML


def plot_figure(piML, mu0ML, mu1ML, sigma0ML, sigma1ML, trainX, trainY ):
	### Plot
	trainX0=np.array([x for x,y in zip(trainX,trainY) if y==0])
	trainX1=np.array([x for x,y in zip(trainX,trainY) if y==1])
	gX0, gX1=np.linspace(-6, 6, 100), np.linspace(-6, 6, 100)
	gX0mesh, gX1mesh = np.meshgrid(gX0, gX1)
	invsig0=np.linalg.inv(sigma0ML)
	invsig1=np.linalg.inv(sigma1ML)
	plt.plot(trainX0[:, 0], trainX0[:,1], 'ro')
	plt.plot(trainX1[:, 0], trainX1[:,1], 'bo')
	levels=[3]
	Z = np.zeros(gX0mesh.shape)
	quad_mat = -.5*(invsig0-invsig1)
	lin_mat = mu0ML.T.dot(invsig0) - mu1ML.T.dot(invsig1)
	const = np.log(piML/(1.-piML)) - np.log(abs(np.linalg.det(sigma0ML)/np.linalg.det(sigma1ML)))-.5*mu0ML.dot(invsig0).dot(mu0ML)+.5*mu1ML.dot(invsig1).dot(mu1ML)
	for i in range(Z.shape[0]):
		for j in range(Z.shape[1]):
			x = np.array([gX0mesh[i,j], gX1mesh[i,j]])
			Z[i, j] = x.T.dot(quad_mat).dot(x) + lin_mat.dot(x) + const
	plt.title("Classification with QDA")
	plt.contour(gX0mesh, gX1mesh, Z, [0])
	plt.show()




def misclassification_error(piML, mu0ML, mu1ML, sigma0ML, sigma1ML, X, Y):
	invsig0=np.linalg.inv(sigma0ML)
	invsig1=np.linalg.inv(sigma1ML)
	quad_mat = -.5*(invsig0-invsig1)
	lin_mat = mu0ML.T.dot(invsig0) - mu1ML.T.dot(invsig1)
	const = np.log(piML/(1.-piML)) - np.log(abs(np.linalg.det(sigma0ML)/np.linalg.det(sigma1ML)))-.5*mu0ML.dot(invsig0).dot(mu0ML)+.5*mu1ML.dot(invsig1).dot(mu1ML)
	Ypred = np.array([1 if x.T.dot(quad_mat).dot(x) + lin_mat.dot(x) + const < 0 else 0 for x in X])
	return 100.*sum(abs(Y-Ypred))/Y.size

#Uncomment this to plot the figure

letter = "C"
plot_bool = True
trainX, trainY = importdata("classification" + letter + ".train")
piML, mu0ML, mu1ML, sigma0ML, sigma1ML = train_model(trainX, trainY)
if plot_bool:
    plot_figure(piML, mu0ML, mu1ML, sigma0ML, sigma1ML, trainX, trainY)

testX, testY = importdata("classification" + letter + ".test")
print "Error on train set:", misclassification_error(piML, mu0ML, mu1ML, sigma0ML, sigma1ML, trainX, trainY), "%"
print "Error on test set:", misclassification_error(piML, mu0ML, mu1ML, sigma0ML, sigma1ML, testX, testY), "%"

