import numpy as np
import matplotlib.pyplot as plt


# Read data file, return data set and label
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('test.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        # Add a column of 1, represent the constant dimension of the model
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


# Sigmoid function
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


# Update model parameters -- Gradient ascent
def gradDescent(dataMatIn, classLabels, lr, maxCycles):
    # Convert input lists to numpy matrix
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        # Calculate sigmoid value of each point
        h = sigmoid(dataMatrix * weights)
        cost = h - labelMat
        weights = weights - lr * dataMatrix.transpose() * cost
    return weights


def plotBestFit(weights):
    dataMat,labelMat=loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def run():
    dataArr, labelMat = loadDataSet()
    lr = 0.2
    maxCycles = 3000
    weights = gradDescent(dataArr, labelMat, lr, maxCycles)
    print(weights)
    plotBestFit(weights.getA())


if __name__ == '__main__':
    run()
