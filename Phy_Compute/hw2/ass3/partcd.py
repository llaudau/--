import numpy as np
import math as mt
import matplotlib.pyplot as plt
import os
from pathlib import Path
script_directory = Path(__file__).parent

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
    return result
def gauss_method(array):
    #gauss method
    def eliminate(array,row):
        array1=array
        rows=array1.shape[0]
        #step1
        divide=array1[row,row]
        array1[row,:]=array1[row,:]/divide
        #step2
        for i in range(row+1,rows):
            divide=array1[i,row]
            array1[i,:]=array1[i,:]-divide*array1[row,:]
        
        return array

    def sequencial_elim(array):
        array1=array
        for i in range(array1.shape[0]):
            array1=eliminate(array1,i)
        return array1
    

    def reverse(array):
        array1=array

        rows=array1.shape[0]
        answer=np.zeros(rows)
        line=rows
        while True:
            if line==0:
                break
            line-=1
            answer[line]=array1[line,-1]
            for i in range(rows-1,line,-1):
                answer[line]-=array1[line,i]*answer[i]
        return answer
    a=sequencial_elim(array)
    return  reverse(a)



def output(x0,y0,out_y0,labe,flag=False):
    if not flag:
        print('reference y : ')
        print(x0)
        print(y0)
    print(labe+': y : ')
    print(out_y0)
    print(labe+': delta y : ')
    print(abs(out_y0-y0))
    if not flag:
        plt.scatter(x0,y0,s=1,label='f')
    plt.scatter(x0,out_y0,s=1,label=labe)
    plt.legend()
    return
#partc


def g(x):
    return abs(x)
def f(x):
    return 1/(1+25 *x**2)

def partc():
    #lag
    flag0=False
    testx2=(np.arange(21)-10)/10
    testy2=g(testx2)
    x3=(np.arange(41)-20)/20
    y3=g(x3)
    Lag_y3=np.zeros(41)
    for i in range(41):
        Lag_y3[i]=Lag(testx2,testy2,x3[i])
    output(x3,y3,Lag_y3,'lag',flag0)
    flag0=True

    #chebshev
    testx3=np.cos(mt.pi*(np.arange(20)+1/2)/(20))
    testy3=g(testx3)
    x4=(testx3[:-1]+testx3[1:])/2
    x4=np.sort(np.concatenate((testx3, x4)))
    y4=g(x4)
    cheb_y4=np.zeros(39)
    for i in range(39):
        cheb_y4[i]=chebyhev(x4[i],testx3,testy3)
    output(x4,y4,cheb_y4,'cheb',flag0)
    plt.ylim(-1,2)
    file_name='partc.png'
    save_path = script_directory / file_name
    plt.savefig(save_path)
    plt.close()
    return
def partd():
    testx=(np.arange(61)-30)/30
    testy=f(testx)
    matrixa=np.zeros((testx.shape[0],testx.shape[0]+1))
    for i in range(1,testx.shape[0]-1):
        matrixa[i,-1]=6*(testy[i-1]/(testx[i]-testx[i-1])/(testx[i+1]-testx[i-1])+testy[i+1]/(testx[i+1]-testx[i])/(testx[i+1]-testx[i-1])-testy[i]/(testx[i+1]-testx[i])/(testx[i]-testx[i-1]))
        matrixa[i,i-1]=(testx[i]-testx[i-1])/(testx[i+1]-testx[i-1])
        matrixa[i,i]=2
        matrixa[i,i+1]=(testx[i+1]-testx[i])/(testx[i+1]-testx[i-1])
    matrixa[0,0]=1
    matrixa[0,-1]=925/4394
    matrixa[testx.shape[0]-1,testx.shape[0]-1]=1
    matrixa[testx.shape[0]-1,-1]=925/4394
    M=gauss_method(matrixa)
    def find_range(x,testx):
        for i in range(testx.shape[0]):
            if testx[i]<=x<=testx[i+1]:
                return i

    def cubic_spline(x,testx,testy,M):
        index=find_range(x,testx)
        # print(index)
        result=-M[index]/6/(testx[index+1]-testx[index])*(x-testx[index+1])**3
        result+=M[index+1]/6/(testx[index+1]-testx[index])*(x-testx[index])**3
        result+=((testy[index+1]-testy[index])/(testx[index+1]-testx[index])-(testx[index+1]-testx[index])/6*(M[index+1]-M[index]))*(x-testx[index])
        result+=testy[index]-M[index]*(testx[index+1]-testx[index])**2/6
        return result
    x0=(np.arange(121)-60)/60
    y0=f(x0)
    cubic_spline_out=np.zeros(x0.shape[0])
    for i in range(x0.shape[0]):
        cubic_spline_out[i]=cubic_spline(x0[i],testx,testy,M)
    output(x0,y0,cubic_spline_out,'cubic_spline')
    file_name='partd.png'
    save_path = script_directory / file_name
    plt.savefig(save_path)
    plt.close()
        
    return


partc()
partd()
