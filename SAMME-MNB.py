
# coding: utf-8

# In[ ]:

## The Multi-class NB (BASE)
import random
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

## Define the base classifier function - MNB with weights
## With train - test set

def resampling(x,y,w):
    [n,d] = x.shape
    xn = [], yn = []
    for k in range(yClassNum):
        weight_classNum = round(weights[np.where(y == yClass[k])].sum()*n,8)
        weight_choice = np.random.choice([np.where(y == yClass[k])], size=weight_classNum, replace=True)
        xk = np.zeros((weight_classNum,d)), yk = np.zeros(weight_classNum)
        xk = x[weight_choice], yk = y[weight_choice]
        xn.append(xk), yn.appened(yk)
    return np.array(xn), np.array(yn)
    
def weightedMNB(xtra,ytra,yClass,yClassNum,w,xt,yt):
    # Resample xTrain & yTrain (x & y) with weight w
    x, y = resampling(x,y,w)
    ## Creating #y*#d dict array
    N_kl = [[dict() for i in range(d)] for j in range(yClassNum)]
    
    for i in range(ntrain):
        yClassTemp = np.argwhere(yClass == y[i])[0,0]  ## Return class order number (0, 1, 2, 3, etc..)
        for j in range(d):
            if round(x[i,j],2) not in N_kl[yClassTemp][j]:
                N_kl[yClassTemp][j][round(x[i,j],2)] = 1
            else:
                N_kl[yClassTemp][j][round(x[i,j],2)] += 1
    
    n_k = np.zeros(yClassNum)
    Pi_k = np.zeros(yClassNum)
    for k in range(yClassNum):
        n_k[k] = np.sum(yTrain == yClass[k])
        Pi_k[k] = (n_k[k]+1.)/float(ntrain+2.)  # yTrain Laplacian correction

    
    ## Weighted output - training data
    yPred = np.zeros(ntrain)
    
    for i in range(ntrain):
        p_xi = np.ones(yClassNum)
        for k in range(yClassNum):
            for j in range(d):
                if round(x[i,j],2) not in N_kl[k][j]:
                    p_xi[k] *= float(1.)/float(n_k[k]+2.)
                else:
                    p_xi[k] *= float(N_kl[k][j][round(x[i,j],2)])/float(n_k[k]+2.)
            p_xi[k] *= Pi_k[k]
        yPred[i] = yClass[np.argmax(p_xi)]
        
    trainDiff = yPred-y
    trainDiff[np.where(trainDiff != 0)] = 1
    trainDiff_n = (trainDiff[np.where(trainDiff != 0)].shape[0])
    train_error = float(trainDiff_n)/ntrain
    
    ## Calculating testing error - testing data
    yTestPred = np.zeros(ntest)  ##n_test

    for i in range(ntest):    ##n_test
        p_xi = np.ones(yClassNum)
        for k in range(yClassNum):
            for j in range(d):            
                if round(xt[i,j],2) not in N_kl[k][j]:
                    p_xi[k] *= float(1.)/float(n_k[k]+2.)
                else:
                    p_xi[k] *= float(N_kl[k][j][round(xt[i,j],2)])/float(n_k[k]+2.)
            p_xi[k] *= Pi_k[k]
        yTestPred[i] = yClass[np.argmax(p_xi)]
    
    #testDiff = yTestPred-y
    #testDiff[np.where(trainDiff == 0)] = 1
    
    return trainDiff, train_error, yTestPred

## Train
yClass = np.unique(np.array(y))
yClassNum = yClass.shape[0]

# Initialize the observation weights using training set
weights = np.ones(ntrain)/float(ntrain)

learnerErrorList = np.zeros(M)  ## corresponding to base trainer error
errorList = np.zeros(M)  ## corresponding to ERR
weightList = np.zeros((M,ntrain))  ## updated weights
alphaList = np.zeros(M)  ## corresponding to alpha

# Set up weak learners number M
M = 600
testWeight = np.zeros((M,ntest))  # For testing data output calculation

for i in range(M):
    errors, nbt_error, testClass = weightedMNB(xTrain,yTrain,yClass,yClassNum,weights,xTest,yTest)
    print nbt_error
    
    e = (errors * weights).sum()/float(weights.sum())   
    alpha = np.log((1-e)/e) + np.log(yClassNum-1)
    
    testWeight[i] = testClass
    learnerErrorList[i] = nbt_error
    weightList[i] = weights
    errorList[i] = e
    alphaList[i] = alpha
    
    ## update weights
    w = np.zeros(ntrain)
    for i in range(ntrain):
        if errors[i] == 1: w[i] = weights[i] * np.exp( alpha/float(yClassNum))   ## Instructor I(c_i != T_m(x_i))
        else: w[i] = weights[i] * np.exp(-alpha * (1.0 - 1.0/float(yClassNum)))      
    weights = w / w.sum()  ## Renormalize 

## For Calculating the output with M Naive Bayes learners together
predP = np.ones((yClassNum,ntest))
TestError = []

for i in range(M):
    for k in range(yClassNum):
        labelOnes = np.ones(ntest)
        labelOnes[np.where(testWeight[i] != yClass[k])] = 0
        predP[k] += alphaList[i] * labelOnes
    Output = yClass[np.argmax(predP, axis = 0)]
    diff_n = (yTest != Output).sum()
    TestError.append(float(diff_n)/ntest)

