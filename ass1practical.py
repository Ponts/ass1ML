import pylab as pb
import numpy as np
from math import pi
import math
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import multivariate_normal
import random


# To sample from a multivariate Gaussian
#f = np.random.multivariate_normal(mu,K)
# To compute a distance matrix between two sets of vectors
#D = cdist ( x1 , x2 )
# To compute the exponetial of all elements in a matrix
#E = np . exp ( D )



def calculateMean(vector):
	sum = 0
	for n in vector:
		sum+=n
	return sum/len(vector)

def calculateStandardVariance(vector):
	sum = 0
	for n in vector:
		sum+= n**2
	return math.sqrt(sum/len(vector))

def plotNormalDist(mean, standVar):
	x, y = np.mgrid[-1:1:.01, -1:1:.01]
	pos = np.empty(x.shape + (2,))
	pos[:, :, 0] = x; pos[:, :, 1] = y
	rv = multivariate_normal(mean, standVar)
	plt.contourf(x, y, rv.pdf(pos))
	plt.show()

def samplePrior(mean, standVar):
	return np.random.multivariate_normal(mean, standVar)

def calcNewMeanStand(mean, standVar, x, t,W):
	newStandVarInv = (1/0.2)*np.linalg.inv(standVar)+(calcY(W,x)**2)
	newMean = np.dot(np.dot(newStandVarInv, np.linalg.inv(standVar)+t*calcY(W,x)),mean)
	return newMean, np.linalg.inv(newStandVarInv)

#ASSIGNMENT 9 STUFF
def calcTi(W,X,e):
	return W[0]*X + W[1] + e
def calcY(W,X):
	return W[0]*X + W[1]
W=np.array([1.5, -0.8])
X=np.arange(-2.0,2.01,0.02)
def sampleError():
	return np.random.normal(0.0,0.2)
def sampleData():
	x = random.choice(X)
	return x, calcTi(W,x,sampleError())

mean = np.array([0.0, 0.0])
standVar=np.identity(2)

plotNormalDist(mean, standVar)
print(mean.shape)
for i in range(10):
	x, t = sampleData()
	W = samplePrior(mean, standVar)
	print(W)
	mean, standVar = calcNewMeanStand(mean,standVar, x, t, W)
	print(mean)
	plotNormalDist(mean,standVar)
