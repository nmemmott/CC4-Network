import numpy as np
from CC4Network import CC4Network


def printPattern(pattern):
    for y in range(0, len(pattern)):
        line = ''
        for x in range(0, len(pattern[y])):
            if pattern[y][x] == 0:
                line = line + " "
            elif pattern[y][x] == 1:
                line = line + "#"
            elif pattern[y][x] == 2:
                line = line + "x"
            elif pattern[y][x] == 3:
                line = line + "o"
        print(line)

def createTrainingSamples(pattern, numberOfSamples):
    printingArray = np.zeros([16,16])
    trainingSamples = np.empty([numberOfSamples,32])
    outputClass = np.empty([numberOfSamples,1])
    for s in range(0, numberOfSamples):
        x = np.random.randint(16)
        y = np.random.randint(16)
        xvector = np.array(list(''.rjust(x+1, '1').rjust(16, '0')), dtype=int)
        yvector = np.array(list(''.rjust(y+1, '1').rjust(16, '0')), dtype=int)
        trainingSamples[s] = np.concatenate((xvector, yvector), axis=0)
        outputClass[s] = pattern[y][x]
        if outputClass[s] == 1:
            printingArray[y,x] = 1
        else:
            printingArray[y,x] = 2
    #printPattern(printingArray)
    return (trainingSamples, outputClass)

    

spiral = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
          [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
          [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
          [0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1],
          [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
          [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
          [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
printPattern(spiral)
(inputSamp, outputSamp) = createTrainingSamples(spiral, 32)
spiralNetwork = CC4Network(inputSamp, outputSamp, 2)
result = np.empty([16,16])
error = 0
for y in range(0,16):
    for x in range(0, 16):
        xvector = np.array(list(''.rjust(x+1, '1').rjust(16, '0')), dtype=int)
        yvector = np.array(list(''.rjust(y+1, '1').rjust(16, '0')), dtype=int)
        coordVector = np.concatenate((xvector, yvector), axis=0)
        if coordVector.tolist() in inputSamp.tolist():
            result[y,x] = 2 if spiral[y][x] == 1 else 3
        else:
            result[y,x] = spiralNetwork.feedForward(coordVector)
            if result[y,x] != spiral[y][x]:
                error+=1

printPattern(result)
print(error)