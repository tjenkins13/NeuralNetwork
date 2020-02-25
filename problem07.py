#!/usr/bin/env python3


from matplotlib import pyplot as plt
import numpy as np
import random
from nn import *
import sys
   
            
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
    if(len(sys.argv)!=2):
        print(f"Correct usage: {sys.argv[0]} <pbm file>")
        sys.exit(0)
    try:
        pic=[]
        with open(sys.argv[1]) as fd:
            fd.readline()
            row,col=list(map(int,fd.readline().strip().split()))
            #print(row,col)
            for line in fd.readlines():
                #print(line)
                if(len(line.strip())>0):
                    pic+=list(map(int,line.strip().split()))
            #print(pic)
    except:
        print(f"Couldn't find {sys.argv[1]}")
        sys.exit(1)
    pic=np.array(pic).reshape((col,row))
        
    shape=[row*col,5]
    n=NeuralNetwork(shape)
    
    n.LoadWeights(['twolevel.npy'])
    plt.imshow(pic)
    label=['H','A','V','X','Y']
    n.TestPic(pic,label)
    plt.show()
