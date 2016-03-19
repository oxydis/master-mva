"""
Author: Valentin Thomas
Script for the LDA maximum likelihood estimation
"""

import numpy as np
import matplotlib.pyplot as plt
from getdata import importdata


def train_model(trainX, trainY):
	trainX0=np.array([x for x,y in zip(trainX,trainY) if y==0])
	trainX1=np.array([x for x,y in zip(trainX,trainY) if y==1])

	### LDA
	piML=1.*trainX1.shape[0]/trainX.shape[0]
	mu0ML=np.mean(trainX0, axis=0)
	mu1ML=np.mean(trainX1, axis=0)
	sigmaML=(sum([np.outer(x0-mu0ML, x0-mu0ML)for x0 in trainX0])
	        +sum([np.outer(x1-mu1ML, x1-mu1ML)for x1 in trainX1]))/trainX.shape[0]
	print "PiML:", piML, "\nmu0ML:", mu0ML, "\nmu1ML:", mu1ML, "\nSigmaML:\n", sigmaML
	return piML, mu0ML, mu1ML, sigmaML


def plot_figure(piML, mu0ML, mu1ML, sigmaML, trainX, trainY ):
	### Plot
	trainX0=np.array([x for x,y in zip(trainX,trainY) if y==0])
	trainX1=np.array([x for x,y in zip(trainX,trainY) if y==1])
	gX0, gX1=np.linspace(-3, 3, 100), np.linspace(-4, 4, 100)
	gX0mesh, gX1mesh = np.meshgrid(gX0, gX1)
	invsig=np.linalg.inv(sigmaML)
	plt.plot(trainX0[:, 0], trainX0[:,1], 'ro')
	plt.plot(trainX1[:, 0], trainX1[:,1], 'bo')
	levels=[3]
	mat=(mu0ML-mu1ML).dot(invsig)
	const = .5*mu0ML.dot(invsig).dot(mu0ML)-.5*mu1ML.dot(invsig).dot(mu1ML)-np.log(piML/(1-piML))
	line = -(mat[0]*gX0 + const)/mat[1]
	plt.title("Classification with LDA")
	plt.plot(gX0, line, label="LDA")
	plt.legend()
	plt.show()



def misclassification_error(piML, mu0ML, mu1ML, sigmaML, X, Y):
	invsig=np.linalg.inv(sigmaML)
	mat=(mu0ML-mu1ML).dot(invsig)
	const = .5*mu0ML.dot(invsig).dot(mu0ML)-.5*mu1ML.dot(invsig).dot(mu1ML)-np.log(piML/(1-piML))
	w = np.array([const, mat[0], mat[1]])
	print w/np.linalg.norm(w)
	Ypred = np.array([1 if mat.dot(x)+const < 0 else 0 for x in X])
	return 100.*sum(abs(Y-Ypred))/Y.size


letter = "A"
plot_bool = False
trainX, trainY = importdata("classification" + letter + ".train")
piML, mu0ML, mu1ML, sigmaML = train_model(trainX, trainY)
if plot_bool:
    plot_figure(piML, mu0ML, mu1ML, sigmaML, trainX, trainY)

testX, testY = importdata("classification" + letter + ".test")
print "Error on train set:", misclassification_error(piML, mu0ML, mu1ML, sigmaML, trainX, trainY), "%"
print "Error on test set:", misclassification_error(piML, mu0ML, mu1ML, sigmaML, testX, testY), "%"





