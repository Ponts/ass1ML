import pylab as pb
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import random

# To sample from a multivariate Gaussian
#f = np.random.multivariate_normal(mu,K)
# To compute a distance matrix between two sets of vectors
#D = cdist ( x1 , x2 )
# To compute the exponetial of all elements in a matrix
#E = np . exp ( D )


def plotNormalDist(mean, coVar):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    x, y = np.mgrid[-5:5:.01, -5:5:0.01]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    rv = multivariate_normal(mean, coVar)
    
    plt.contourf(x, y, rv.pdf(pos))
    plt.show()

def getCoVar(X):
    return np.linalg.inv(np.dot(X.T,X) + np.identity(2))

def getMean(coVar, X,T):
    return np.array(np.dot(coVar, np.dot(X.T,T))).reshape(-1)

def calculateNewParams(x,t,sigma):
	#newCoVar = np.linalg.inv(np.dot(x.T,x) + np.identity(2))
	#newMean = np.array(np.dot(sigma, np.dot(x.T,t))).reshape(-1)
	newCoVar = getCoVar(x)
	newMean = getMean(newCoVar,x,t)
	return newMean, newCoVar

def sampleWeights(mean, covar):
	return np.random.multivariate_normal(mean,covar)

truW = [1.5, -0.8]

x = np.arange(-2,2,.02)
x = np.expand_dims(x,axis=1)
x = np.insert(x,1,1,axis=1) 





trainX = np.matrix([random.choice(x) for i in range(20)])
trainT = (np.dot(trainX, truW) + np.random.normal(0,0.2,trainX.shape[0])).T

sigma = 1

newMean, newCoVar = calculateNewParams(trainX,trainT,sigma)
plotNormalDist(newMean,newCoVar)

for i in range(5):
	W = sampleWeights(newMean,newCoVar)
	
	plt.plot(x[:,0],np.dot(x,W))
plt.scatter(np.squeeze(np.asarray(trainX[:,0])),np.squeeze(np.asarray(trainT)))
plt.show()
