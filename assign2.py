# Data Mining Fall 2019 Assignment 2
# Mario Formisano

import sys, os
import numpy as np
import pandas as pd
import matplotlib as plt


def projection(a, b):
    # projection of a onto b (a in the b direction)
    # proj b (a)
    # return np.dot(b, ( np.dot(a, b) / np.dot(b, b) ) )
    return np.dot(a.T, b) / np.dot(b.T, b)

def QR(augData, qMatrix, rMatrix):
    # Find Q (matrix of U vectors)
    ''' Output Test Initializations for Hydrogen Plugin
    trainData = np.loadtxt(trainFile, delimiter=',')
    lastindex = trainData.shape[1] - 1
    trainData = np.delete(trainData, [lastindex], 1)
    augData = np.concatenate((np.ones((n,1)), trainData), axis=1)
    dPlusOne = augData.shape[1]
    n = augData.shape[0]
    qMatrix = np.empty(shape=(n, dPlusOne), dtype=float)
    rMatrix = np.eye(dPlusOne, dtype=float)'''

    # U0 is A0 which is our column of ones.
    u0 = augData[:,0]
    qMatrix[:,0] = u0
    # Generate U1 - Ud:
    # Ud = Ad - Pd0 * U0 - Pd1 * U1 - ... - Pdd-1 * Ud-1
    dPlusOne = augData.shape[1]
    for i in range(1, dPlusOne):
        projectionSum = u0
        for j in range(i):
            pij = projection(augData[:,i], qMatrix[:,j])
            rMatrix[j, i] = pij
            projectionSum = projectionSum + np.dot(pij, qMatrix[:,j])
        ui = augData[:,i] - projectionSum
        qMatrix[:,i] = ui

    # R is the set of projections of Aj onto Ui.
    # Q is our orthogonal basis.

def backsolve(Q, R, y):
    delta = np.dot(Q.T, Q)
    B = np.dot(np.linalg.inv(delta), np.dot(Q.T, y))
    w = np.dot(np.linalg.inv(R), B)
    return w

def normL2(w, d):
    # Consider: d should be unaugmented d, not 'd + 1'
    sum = 0
    for i in range(d):
        sum = sum + (w[i] ** 2)
    return sum

def tss(y):
    # Total scatter of response variable
    m = np.mean(y)
    n = y.shape[0]
    sum = 0
    for i in range(n):
        sum = sum + ((y[i] - m) ** 2)
    return sum

def prepData(filename):
    # Prepare data, augmented data matrix, y, n, and d for use in main()
    data = np.loadtxt(filename, delimiter=',')
    lastindex = data.shape[1] - 1
    # Last field is the response variable
    y = data[:,lastindex]
    data = np.delete(data, [lastindex], 1)
    n = data.shape[0]
    d = data.shape[1]
    # augment the data: Add A0 to Matrix
    augData = np.concatenate((np.ones((n,1)), data), axis=1)
    return data, augData, y, n, d

def main():
    sys.stdout = open("assign2.txt", "w")
    trainFile = "train.txt"
    testFile = "test.txt"
    # 4390 input
    if len(sys.argv) == 3:
        trainFile = sys.argv[1]
        testFile = sys.argv[2]

    # Initialize matrices
    trainData, augData, y, n, d = prepData(trainFile)

    # Initialize Q and R
    dPlusOne = augData.shape[1]
    qMatrix = np.empty(shape=(n, dPlusOne), dtype=float)
    rMatrix = np.eye(dPlusOne, dtype=float)

    # Populate Q and R via QR Factorization.

    QR(augData, qMatrix, rMatrix)

    # We have Q and R.  Now find augmented weight vector w via
    # back substitution.

    w = backsolve(qMatrix, rMatrix, y)
    print("1. The weight vector w:")
    print(w, "\n")

    # Now find L2 norm of the weight vector w.

    L2 = normL2(w, d)
    print("2. The L2 norm of the weight vector:")
    print(L2, "\n")

    # Calculate SSE and R^2 values on the training data.

    yhat = np.dot(augData, w)
    trainSSE = np.dot((y - yhat).T, (y - yhat))
    trainTSS = tss(y)
    r2 = (trainTSS - trainSSE) / trainTSS
    # r2 should approach 1.  It does!

    print("3. SSE and R^2 values of the training data:")
    print("SSE:", trainSSE)
    print("R^2:", r2, "\n")

    # Set up the testing data, and calculate SSE and R^2.

    testData, augTest, yTest, nTest, dTest = prepData(testFile)
    yhatTest = np.dot(augTest, w)
    testSSE = np.dot((yTest - yhatTest).T, (yTest - yhatTest))
    testTSS = tss(yTest)
    testR2 = (testTSS - testSSE) / testTSS

    print("4. SSE and R^2 values of the testing data:")
    print("SSE:", testSSE)
    print("R^2:", testR2)

if __name__ == "__main__":
    main()
