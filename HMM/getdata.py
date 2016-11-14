import numpy as np


def importdata(filename):
    data = np.loadtxt(filename)
    trainX = data[:, 0:2]
    trainY = data[:, -1]
    return trainX, trainY
