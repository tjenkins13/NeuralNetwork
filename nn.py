#!/usr/bin/env python3


from matplotlib import pyplot as plt
import numpy as np
import random

class NeuralNetwork:
    def __init__(self,shape,activator="sigmoid",batch=1,alpha=.1):
        if len(shape)<2:
            print("Error: must have at least an input and output layer")
        self.layers=len(shape)
        self.shape=shape
        self.alpha=alpha
        self.weights=[]
        self.batch=1
        for i in range(self.layers-1):
            self.weights.append(np.random.randn(self.shape[i]*self.shape[i+1]).reshape((self.shape[i+1],self.shape[i])))
            
        #numweights=sum([shape[i]*shape[i+1] for i in range(len(shape)-1)])
        #weights=np.random.randn(numweights).reshape(tuple([len(shape)]+[shape[i+1],shape[i] for i in range(len(shape)-1)]))
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def sigmoid_d(self,x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def feedforward(self):
        self.z=[]
        self.a=[self.data]
        #self.a=[]
        for i in range(self.layers-1):
            #print("in Feed forward")
            #print(self.a[-1].shape,self.weights[i].shape)
            #self.z.append(np.matmul(self.a[-1],self.weights[i]))
            self.z.append(np.matmul(self.weights[i],self.a[-1]))
            self.a.append(self.sigmoid(self.z[-1]))

        '''for layer in self.weights[1:]:
            self.a.append(self.sigmoid(self.z[-1]))
            self.z.append(np.matmul(self.a[-1],layer))
            '''

    def backprop(self): #possible change to alter alpha later
        self.delta=[(self.a[-1]-self.label)*self.a[-1]*(1-self.a[-1])]
        for i in range(self.layers-2,0,-1):
            #print("in backprop")
            #print(self.weights[i].shape,self.delta[0].shape,self.a[i].shape)
            self.delta=[np.matmul(self.weights[i].T,self.delta[0])*self.a[i]*(1-self.a[i])]+self.delta

        for i in range(len(self.weights)):
            #print("hello")
            print(self.weights[i].shape,self.delta[i].shape,self.a[i].shape,np.outer(self.delta[i],self.a[i]).shape)
            self.weights[i]-=self.alpha*np.outer(self.delta[i],self.a[i])

    def Cost(self):
        return .5*sum(self.a[-1]**2-self.label**2)

    def Train(self,data,label):
        self.data=data
        self.label=label
        self.feedforward()
        self.backprop()

    def TrainLoop(self,training_data,num_iter): #update later with tqdm and allow batch support
        for iter in range(num_iter):
            category=random.randrange(len(training_data))
            item=random.randrange(len(training_data[category]))
            pic=training_data[category][item].reshape(-1,1)
            label=np.zeros(len(training_data)).reshape(-1,1)
            label[category]=1
            self.Train(pic,label)


    def TestPic(self,pic,data_label):
        self.data=pic.reshape(-1,1)
        self.feedforward()
        print(self.a[-1])
        maxv=max(self.a[-1])
        ccat=list(self.a[-1]).index(maxv)
        print(max(self.a[-1]))
        print(f"{data_label[ccat]} was chosen")
        return (maxv,data_label[ccat])

    def Test(self,testing_data,data_label=['H','A','V','X','Y']):
        totalright=0
        total=0
        for category in range(len(testing_data)):
            numright=0
            totalc=0
            for pic in testing_data[category]:
                self.data=pic.reshape(-1,1)
                self.feedforward()
                print(self.a[-1])
                maxv=max(self.a[-1])
                ccat=list(self.a[-1]).index(maxv)
                print(max(self.a[-1]))
                print(f"{data_label[ccat]} was chosen when the correct answer is {data_label[category]}")
                if(ccat==category):
                    numright+=1
                totalc+=1
            print(f"We correctly identified {numright*100/totalc}% of {data_label[category]}'s")
            total+=totalc
            totalright+=numright
        print(f"Overall, we correctly identified {totalright*100/total}% of the pictures")
                
    def SaveWeights(self,files):
        if len(files)!=len(self.weights):
            print(f"You need to have a list of {len(self.weights)} files to save all of your weights")
            return
        for i in range(len(self.weights)):
            np.save(files[i],self.weights[i])

    def LoadWeights(self,files):
        a=np.load(files[0])
        self.weights=[a]
        self.shape=[a.shape[1],a.shape[0]]
        for file in files[1:]:
            a=np.load(file)
            self.weights.append(a)
            self.shape.append(a.shape[0])
        self.layers=len(self.shape)
            

            
def get_pic(filename,ncol,nrow):
    with open(filename) as fd:
        magic=fd.readline()
        cols,rows=list(map(int,fd.readline().strip().split(' ')))
        #maxval=int(fd.readline().strip())
        pic=[]
        for line in fd.readlines():
            if(len(line.strip())>0):
                l=list(map(int,line.strip().split(' ')))
                pic+=l
    
    pcol=cols//ncol
    prow=rows//nrow

    '''bwidth=0
    bheight=0
    
    nparr=np.array(pic)
    nparr=nparr.reshape((rows,cols))
    wborder=[nparr[0][0]]*len(nparr[0])
    for i in nparr:
        if list(i)==wborder:
            bheight+=1
        else:break

    bwidth=bheight
    '''
    nparr=np.array(pic).reshape(nrow, prow, ncol, pcol).swapaxes(1,2).reshape(-1, prow, pcol) #if you leave out last reshape you can give row,col to get pic
    #nparr=np.array(list(map(lambda x: x[bheight:len(x)-bheight,bwidth:len(x[0])-bwidth],nparr)))
    #nparr=nparr.reshape((len(nparr)//5,5,prow,pcol)).swapaxes(0,1)

    return nparr
        
if __name__=='__main__':
    training_data=get_pic("../hafxytrain.pbm",54,20)
    training_data=training_data.reshape((len(training_data)//5,5,48,32)).swapaxes(0,1)
    test_data=get_pic("../hafxytest.pbm",32,10)
    testing_data=test_data.reshape((5,-1,48,32))
    shape=[training_data.shape[-1]*training_data.shape[-2],60,5]
    n=NeuralNetwork(shape)
    n.TrainLoop(training_data,10000)
    n.Test(testing_data)
    n.SaveWeights(['w1','w2'])
    plt.imshow(training_data[0][0])
    label=['H','A','V','X','Y']
    plt.show()
