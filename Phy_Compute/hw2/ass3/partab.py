import numpy as np
import math as mt
import matplotlib.pyplot as plt
import os
from pathlib import Path
script_directory = Path(__file__).parent


#part(a)
testx=(np.arange(21)-10)/10
def f(x):
    return 1/(1+25 *x**2)

testy=f(testx)


x0=(np.arange(41)-20)/20
y0=f(x0)
Lag_y0=np.zeros(41)
def Lag(testx,testy,x):

    def L(x,i,testx):
        result=1
        for j in range(testx.shape[0]):
            if j==i:
                continue
            result=result*(x-testx[j])/(testx[i]-testx[j])
        return result

    def lagran(x,testx,testy):
        result=0
        for i in range(testx.shape[0]):
            result+=testy[i]*L(x,i,testx)
        return result
    return lagran(x,testx,testy)
for i in range(41):
    Lag_y0[i]=Lag(testx,testy,x0[i])

def output(x0,y0,out_y0,part):
    print(f'{part} x :')
    print(x0)
    print(f'{part} f(x) :')
    print(y0)
    print(f'{part} P20(x) :')
    print(out_y0)
    print(f'{part} |P20(x) - f(x)|:')
    print(abs(out_y0-y0))
    
    plt.scatter(x0,y0,s=1,label='f')
    plt.scatter(x0,out_y0,s=1,label='P20')
    plt.legend()
    filename=part+'.png'
    path=script_directory/filename
    plt.savefig(path)
    plt.close()
    return
output(x0,y0,Lag_y0,'parta')
#partb
testx1=np.cos(mt.pi*(np.arange(20)+1/2)/(20))
testy1=f(testx1)
# print(x0.shape)
def chebyhev_para(n,testx1,testy1):
    c=0
    N=testx1.shape[0]
    for i in range(N):
        c+=mt.cos(n*mt.acos(testx1[i]))*testy1[i]
    if n==0:
        c=c/N
    else:
        c=c*2/N
    return c
def chebyhev(x,testx1,testy1):
    N=testx1.shape[0]
    result=0
    for i in range(0,N+1):
        result+=chebyhev_para(i,testx1,testy1)*mt.cos(i*mt.acos(x))
    return result#-chebyhev_para(0,testx1,testy1)/2

x1=(testx1[:-1]+testx1[1:])/2
x1=np.sort(np.concatenate((testx1, x1)))
y1=f(x1)
cheb_y1=np.zeros(39)
for i in range(39):
    cheb_y1[i]=chebyhev(x1[i],testx1,testy1)
output(x1,y1,cheb_y1,'partb')

