import pandas as pd
import random
import numpy as np
import pickle
def loadDict():
    voca_path='data/vocab.pickle'
    with open(voca_path,'rb') as fr:
        vocab = pickle.load(fr)
    id2word, word2id={},{}
    for id,word in enumerate(vocab):
        id2word[id]=word
        word2id[word]=id
    return id2word,word2id
def sentense2id(input,word2id,paddLen=20):
    sentenseEmb=[[word2id.get(word,word2id['UNK']) for word in sentense]for sentense in input]
    for i in range(len(sentenseEmb)):
        if len(sentenseEmb[i])>=paddLen:
            sentenseEmb[i]=sentenseEmb[i][:paddLen]
        else:
            sentenseEmb[i] =sentenseEmb[i]+[0]*(paddLen-len(sentenseEmb[i]))
    return sentenseEmb

class genData(object):
    def __init__(self,rate=0.7,shuffle=True,path='data/trainData.csv'):
        self.path = path
        self.data = pd.read_csv(path)
        id2word, word2id = loadDict()
        X = np.array(self.data['title'].tolist())
        self.X=np.array(sentense2id(X, word2id))
        self.Y = np.array(self.data['calssId'].tolist())
        if shuffle:
            shuffleIndex=list(range(len(self.data)))
            random.shuffle(shuffleIndex)
            self.X = self.X[shuffleIndex]
            self.Y = self.Y[shuffleIndex]
        index = int(len(self.data)*rate)
        self.xtrain = self.X[:index]
        self.ytrain = self.Y[:index]
        self.xtest = self.X[index:]
        self.ytest = self.Y[index:]

    def returnData(self,batchSize):
        times1 = len(self.xtrain) // batchSize
        times2 = len(self.xtest) // batchSize
        xtrain, ytrain, xtest, ytest=[],[],[],[]
        for i in range(times1):
            begin = i*batchSize
            end = (i+1)*batchSize
            xtrain.append(self.xtrain[begin:end])
            ytrain.append(self.ytrain[begin:end])
        for i in range(times2):
            begin = i*batchSize
            end = (i+1)*batchSize
            xtest.append(self.xtest[begin:end])
            ytest.append(self.ytest[begin:end])
        return xtrain, ytrain,xtest, ytest

    def returnXY(self):
        return self.X,self.Y

    def genTrainIter(self,batchSize):
        times = len(self.xtrain)//batchSize
        for i in range(times):
            begin = i*batchSize
            end = (i+1)*batchSize
            yield (self.xtrain[begin:end],self.ytrain[begin:end])
    def genTestIter(self,batchSize):
        times = len(self.xtest) // batchSize
        for i in range(times):
            begin = i*batchSize
            end = (i+1)*batchSize
            yield (self.xtest[begin:end],self.ytest[begin:end])
# dd= genData()
# aa=dd.genIter(10)
# for i in aa:
#     print(i)