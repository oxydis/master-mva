import numpy as np
from getdata import importdata
import LDA as lda
import logistic_regression as log_reg
import linear_regression as lin_reg
import QDA as qda

letters = ["A", "B", "C"]
scores = np.zeros((4, 6)) #lines: algorithms, columns: data sets

for i, letter in enumerate(letters):
    trainX, trainY = importdata("classification"+letter+".train")
    testX, testY = importdata("classification"+letter+".test")
    w_log = log_reg.train_model(trainX, trainY)
    w_lin = lin_reg.train_model(trainX, trainY)
    pi, mu0, mu1, sigma = lda.train_model(trainX, trainY)
    pi, mu0, mu1, sigma0, sigma1 = qda.train_model(trainX, trainY)
    scores[0, 2*i] = lda.misclassification_error(pi, mu0, mu1, sigma, trainX, trainY)
    scores[0, 2*i+1] = lda.misclassification_error(pi, mu0, mu1, sigma, testX, testY)
    scores[1, 2*i] = log_reg.misclassification_error(w_log, trainX, trainY)
    scores[1, 2*i+1] = log_reg.misclassification_error(w_log, testX, testY)
    scores[2, 2*i] = lin_reg.misclassification_error(w_lin, trainX, trainY)
    scores[2, 2*i+1] = lin_reg.misclassification_error(w_lin, testX, testY)
    scores[3, 2*i] = qda.misclassification_error(pi, mu0, mu1, sigma0, sigma1, trainX, trainY)
    scores[3, 2*i+1] = qda.misclassification_error(pi, mu0, mu1, sigma0, sigma1, testX, testY)

print "The scores are:\n", scores
