
# coding: utf-8

# In[ ]:

## The Multi-class NB (BASE)
import random
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

def MNB(xTrain,yTrain,yClass,yClassNum):   
    # Creating #y*#d dict array, for event counts recording
    N_kl = [[dict() for i in range(d)] for j in range(yClassNum)]

    for i in range(ntrain):
        yClassTemp = np.argwhere(yClass == yTrain[i])[0,0]  # Return class number
        for j in range(d):
            if round(xTrain[i,j],2) not in N_kl[yClassTemp][j]:
                N_kl[yClassTemp][j][round(xTrain[i,j],2)] = 1
            else:
                N_kl[yClassTemp][j][round(xTrain[i,j],2)] += 1

    n_k = np.zeros(yClassNum)
    Pi_k = np.zeros(yClassNum)
    for k in range(yClassNum):
        n_k[k] = np.sum(yTrain == yClass[k])
        Pi_k[k] = (n_k[k]+1.)/float(ntrain+2.)  # yTrain Laplacian correction
    
    return N_kl, n_k, Pi_k

random.seed(40) #Seed the random number generator
## A. Iris for instance
# Load data and utilize random permutation
z = np.genfromtxt('iris.data',dtype='str',delimiter=',')
rp = np.random.permutation(z.shape[0])
z = z[rp]

# Take 2/3 as training set, 1/3 as testing set
y = np.array([ord(char.lower()) - 96 for char in z[:,0]])
X = np.array(z[:,1:],dtype=float)
[n,d] = X.shape
ntrain = int(round(n*2./3.)) 
ntest = n - ntrain
xTrain, xTest = X[:ntrain], X[ntrain:]
yTrain, yTest = y[:ntrain], y[ntrain:]

# Train (via event counts, i.e. discrete prior probability assumption)
yClass = np.unique(np.array(yTrain))
yClassNum = yClass.shape[0]

N_kl, n_k, Pi_k = MNB(xTrain,yTrain,yClass,yClassNum)

# Test
yPred = np.zeros(ntest)  ##n_test
for i in range(ntest):    ##n_test
    p_xi = np.ones(yClassNum)
    for k in range(yClassNum):
        for j in range(d):
            if round(xTest[i,j],2) not in N_kl[k][j]:
                p_xi[k] *= float(1.)/float(n_k[k]+2.)
            else:
                p_xi[k] *= float(N_kl[k][j][round(xTest[i,j],2)]+1.)/float(n_k[k]+2.)
        p_xi[k] *= Pi_k[k]
    yPred[i] = yClass[np.argmax(p_xi)]

# Compare prediction (output) with yTest
diff_n = (yPred != yTest).sum()
test_error = float(diff_n)/ntest
print "Number of error test: {}".format(diff_n)
print "Test Error: %.2f%%" % (test_error * 100)

