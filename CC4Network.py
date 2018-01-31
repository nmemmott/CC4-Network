import numpy as np

class CC4Network:
    
    def _hammingWeight(self, vector):
        hw = 0
        for i in range(0, len(vector)):
            if vector[i] == 1:
                hw+=1
        return hw
    
    def __init__(self, trainIn, trainOut, radius):
        #Create weights
        hiddenSize = len(trainIn)
        if hiddenSize==0:
            raise ValueError('There must be at least one training sample.')
        inputSize = len(trainIn[0])
        outSampLen = len(trainOut)
        if hiddenSize != outSampLen:
            raise ValueError('There must be as many output vectors as input vectors.')
        outputSize = len(trainOut[0])
        self.inWeights = np.empty([inputSize+1, hiddenSize])
        self.outWeights = np.empty([hiddenSize, outputSize])
        for h in range(0, hiddenSize):
            s = self._hammingWeight(trainIn[h])
            self.inWeights[inputSize, h] = radius - s + 1
            for i in range(0, inputSize):
                if trainIn[h][i] == 1:
                    self.inWeights[i,h] = 1
                else:
                    self.inWeights[i,h] = -1
            for o in range(0, outputSize):
                if trainOut[h][o] == 1:
                    self.outWeights[h, o] = 1
                else:
                    self.outWeights[h, o] = -1
        
        #Create activation function
        self.activationFunction = np.vectorize(lambda x: 1 if x>0 else 0)
    
    
    def feedForward(self, input):
        input = np.append(input, [1])
        return self.activationFunction(np.matmul(self.activationFunction(np.matmul(input, self.inWeights)), self.outWeights))
