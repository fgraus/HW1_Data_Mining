import time
from crossValidation import mainCrossValidation
from computeSimilarity import returnPredict
from getData import getDataToWork

def main():
    testProgram()

def testProgram():
    trainList, testList = getDataToWork()

    file = open('Resources/pruebas.dat','w')
    timeStarted = time.time()
    i = 0
    for test in testList:
        timeNow = time.time()
        i += 1
        vel = (timeNow - timeStarted) / i
        tRestante = (len(testList)-i)*vel
        print('Minutes left ', tRestante/60)

        guess = returnPredict(trainList, test)
        if (guess == 1):
            file.write('+1\n')
        else:
            file.write('-1\n')
    file.close()

def checkProgram():
    timeNow = time.time()
    mediumAccuracy = mainCrossValidation()
    print('Final accuracy is %.4f'%mediumAccuracy, ' in a time of ', time.time()-timeNow)


main()
