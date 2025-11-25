import numpy as np
import math as mt
import matplotlib.pyplot as plt
import os
from pathlib import Path
script_directory = Path(__file__).parent


#part(a)

def f(x):
    return np.sin(2*mt.pi*x)/(1+x**2)

def g(x):
    return 3/2*x+7/2
def output(x0,y0,out_y0,order):
    # print(f'{part} x :')
    # print(x0)
    # print(f'{part} f(x) :')
    # print(y0)
    # print(f'{part} P20(x) :')
    # print(out_y0)
    # print(f'{part} |P20(x) - f(x)|:')
    # print(abs(out_y0-y0))
    if order==5:
        plt.scatter(x0,y0,s=1,label='f')
    plt.scatter(x0,out_y0,s=1,label=f'order = {order}')
    # plt.legend()
    # plt.show()
    # filename=part+'.png'
    # path=script_directory/filename
    # plt.savefig(path)
    # plt.close()
    return


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

for N in [5,10,15]:
    testx1=np.cos(mt.pi*(np.arange(N)+1/2)/(N))
    testy1=f(g(testx1))
    x1=(np.arange(500)-250)/500
    y1=f(g(x1))
    cheb_y1=np.zeros(500)
    for i in range(500):
        cheb_y1[i]=chebyhev(x1[i],testx1,testy1)
    output(g(x1),y1,cheb_y1,N)
plt.legend()
# plt.show()
filepath=script_directory/"ass1.png"
plt.savefig(filepath,dpi=300)