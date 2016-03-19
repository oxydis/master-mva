## Linear Regression

import numpy as np
from getdata import importdata
import matplotlib.pyplot as plt



def train_model(trainX, trainY):
    Y = 1*trainY - 0
    col1 = np.ones((trainX.shape[0], 1))
    # We add the bias into the X vector for simplicity
    X = np.append(col1, trainX, axis = 1)
    w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    print "Final estimation for w:", w
    return w

def plot_figure(w, trainX, trainY):
    ### Plot
    trainX0=np.array([x for x,y in zip(trainX,trainY) if y==0])
    trainX1=np.array([x for x,y in zip(trainX,trainY) if y==1])
    gX = np.linspace(-2, 2, 100, endpoint=True)
    plt.plot(trainX0[:, 0], trainX0[:,1], 'ro')
    plt.plot(trainX1[:, 0], trainX1[:,1], 'bo')
    line=(.5-w[1]*gX - w[0])/w[2]
    plt.title("Classification with Linear Regression")
    plt.plot(gX, line)
    plt.show()

def misclassification_error(w, X, Y):
    col1 = np.ones((X.shape[0], 1))
    X = np.append(col1, X, axis = 1)
    mat = (X.dot(w)).ravel()
    Ypred = np.array([1 if ypred > .5 else 0 for ypred in mat])
    return 100.*sum(abs(Y-Ypred))/Y.size

letter = "A"
plot_bool = False
trainX, trainY = importdata("classification" + letter + ".train")
w = train_model(trainX, trainY)
if plot_bool:
    plot_figure(w, trainX, trainY)

testX, testY = importdata("classification" + letter + ".test")
print "Error on train set:", misclassification_error(w, trainX, trainY), "%"
print "Error on test set:", misclassification_error(w, testX, testY), "%"
