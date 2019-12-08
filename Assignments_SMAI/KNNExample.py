# Example of kNN implemented from Scratch in Python

import csv
import random
import math
import operator
import numpy as np
import sys

def loadDataset(filename, split, trainingSet=[] , testSet=[]):
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])

def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    num_points = len(lines)
    dim_points = 28 * 28
    data = np.empty((num_points, dim_points))
    labels = np.empty(num_points)

    for ind, line in enumerate(lines):
        num = line.split(',')
        labels[ind] = int(num[0])
        data[ind] = [int(x) for x in num[1:]]

    return (data, labels)


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    print(distances)
    return neighbors

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

def main():
    # prepare data
    trainingSet=[]
    testSet=[]
    split = 0.67
    #loadDataset('C:\\Users\\suagrawa\\Desktop\\Spring_2019_IIIT\\Monsoon 2019\\SMAI Assignments\\Assignment-1\\Anomaly-Detection-master\\train.csv', split, trainingSet, testSet)
    trainingSet,train_y = read_data('C:\\Users\\suagrawa\\Desktop\\Spring_2019_IIIT\\Monsoon 2019\\SMAI Assignments\\Assignment-1\\Anomaly-Detection-master\\train1.csv')
    testSet,test_y = read_data('C:\\Users\\suagrawa\\Desktop\\Spring_2019_IIIT\\Monsoon 2019\\SMAI Assignments\\Assignment-1\\Anomaly-Detection-master\\test1.csv')

    print(test_y)
    #sys.exit(0)
    print('Train set: ' + repr(len(trainingSet)))
    print('Test set: ' + repr(len(testSet)))
    # generate predictions
    predictions=[]
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        print(neighbors)
        result = getResponse(neighbors)
        predictions.append(result)
        print(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(test_y[x]))
    accuracy = getAccuracy(test_y, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')

main()