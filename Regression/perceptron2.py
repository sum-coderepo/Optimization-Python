import sys

fileName = "C:\\Users\\suagrawa\\Optimization-Python\\Regression\\input"
data = []
def readFromFile(fileName):
    with open(fileName) as f:
        content = f.readlines()
        content = [x.strip() for x in content]

        for item in content:
            row = [int(el) for el in item.split(',')]
            data.append(row)
    return data

def sign(row, weights):
    firstValue = row[0] * weights[0]
    secondValue = row[1] * weights[1]
    sum = weights[2] + firstValue + secondValue
    return 1 if sum >= 0 else -1

def perceptronAlgorithm(data):
    weights = [0 for i in range(len(data[0]))]
    result = ""
    while True:
        isFinal = True
        for i in range(0, len(data)):
            expected = data[i][2]
            predicted = sign(data[i], weights)
            if expected * predicted <= 0:
                isFinal = False
                weights[0] = weights[0] + expected * data[i][0]
                weights[1] = weights[1] + expected * data[i][1]
                weights[2] = weights[2] + expected

        if isFinal:
            result += str(weights[0]) + ", "  + str(weights[1]) + ", " + str(weights[2])
            break
        else:
            result += str(weights[0]) + ", "  + str(weights[1]) + ", " + str(weights[2]) + "\n"


def writeToFile(result):
    outputFileName = sys.argv[2]
    f = open(outputFileName, 'w')
    f.write(result)
    f.close()

data = readFromFile(fileName)
print(data)
result = perceptronAlgorithm(data)
print(result)
#writeToFile(result)