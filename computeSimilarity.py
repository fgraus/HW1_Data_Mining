import numpy as np
import constants

def returnPredict(train, test):
    return(guessResult(returnNeighbords(test, train)))

def computeAccuracy(train, testList):

    correctOnes = 0

    for test in testList:
        
        vec = returnNeighbords(test, train)

        if (guessResult(vec) == test[0]):
            correctOnes += 1

    return correctOnes / len(testList)

def returnNeighbords(element, trainList):

    NUMBERNEIGBOURDS = constants.neighbourds


    bestNeigbours = np.zeros((NUMBERNEIGBOURDS,2))

    minSim = 0
    posMin = 0

    for train in trainList:

        sim = returnSimilarity(train, element)

        if (sim > minSim):
        
            bestNeigbours[posMin][0] = sim
            bestNeigbours[posMin][1] = train[0]
            
            posMin = np.argmin(bestNeigbours)
            minSim = bestNeigbours[posMin][0]

    return bestNeigbours

def returnSimilarity(train, test):

    # I am gonna use cosine similarity because it is an sparce data

    cosSimilarity = np.dot(test[1], train[1])

    return cosSimilarity

def guessResult(bestNeigbourds):
    
    #Here we are gonna try what mayority say but we can check others methods
    result = 0
    for voter in bestNeigbourds:
        result += voter[1]

    if result >= 0:
        return 1
    else:
        return -1

