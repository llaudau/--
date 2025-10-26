import numpy as np
import math as mt
import matplotlib.pyplot as plt
import os
from pathlib import Path
script_directory = Path(__file__).parent

# we have several x sets under:
# Lebesgue x 20 points
# uniform x 21 points
# uniform x 61 points


def L(x,i,testx):
    result=1
    for j in range(testx.shape[0]):
        if j==i:
            continue
        result=result*(x-testx[j])/(testx[i]-testx[j])
    return result

def Lebfunc(x,testx):
    res=0
    for i in range(testx.shape[0]):
        res+=L(x,i,testx)
    return res
def parte():
    Lebx=np.cos(mt.pi*(np.arange(20)+1/2)/(20))
    unix21=(np.arange(21)-10)/10
    unix61=(np.arange(61)-30)/30
    outx=np.linspace(-1,1,1000)
    Leb=np.zeros(outx.shape[0])
    uni21=np.zeros(outx.shape[0])
    uni61=np.zeros(outx.shape[0])
    for i in range(outx.shape[0]):
        Leb[i]=Lebfunc(outx[i],Lebx)
        uni21[i]=Lebfunc(outx[i],unix21)
        uni61[i]=Lebfunc(outx[i],unix61)
    plt.scatter(outx,Leb,s=1,label='Leb')
    plt.scatter(outx,uni21,s=1,label='uni21')
    plt.scatter(outx,uni61,s=1,label='uni61')
    plt.legend()
    name='parte.png'
    path=script_directory/name
    # plt.ylim(0.99,1.01)
    plt.savefig(path)
    return
parte()