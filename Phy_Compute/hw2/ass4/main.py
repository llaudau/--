import numpy as np
import math as mt
import matplotlib.pyplot as plt
import os
from pathlib import Path
script_directory = Path(__file__).parent

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


#parta
t=np.arange(9)
phi=np.arange(9)*mt.pi/4
x=(1-np.cos(phi))*np.cos(phi)
y=(1-np.cos(phi))*np.sin(phi)
print('x : ')
print(x)
print('y : ')
print(y)

#partb
def S(testx,testy):
    
    matrixa=np.zeros((testx.shape[0],testx.shape[0]+1))
    for i in range(1,testx.shape[0]-1):
        matrixa[i,-1]=6*(testy[i-1]/(testx[i]-testx[i-1])/(testx[i+1]-testx[i-1])+testy[i+1]/(testx[i+1]-testx[i])/(testx[i+1]-testx[i-1])-testy[i]/(testx[i+1]-testx[i])/(testx[i]-testx[i-1]))
        matrixa[i,i-1]=(testx[i]-testx[i-1])/(testx[i+1]-testx[i-1])
        matrixa[i,i]=2
        matrixa[i,i+1]=(testx[i+1]-testx[i])/(testx[i+1]-testx[i-1])
    matrixa[0,0]=1
    matrixa[0,-2]=-1
    matrixa[testx.shape[0]-1,0]=(testx[0]-testx[1])**2/(testx[1]-testx[0])/2-(testx[1]-testx[0])/6
    matrixa[testx.shape[0]-1,-2]=(testx[-1]-testx[-2])**2/(testx[-1]-testx[-2])/2-(testx[-1]-testx[-2])/6
    matrixa[testx.shape[0]-1,1]=(testx[1]-testx[0])/6
    matrixa[testx.shape[0]-1,-3]=(testx[-1]-testx[-2])/6
    matrixa[testx.shape[0]-1,-1]=(testy[1]-testy[0])/(testx[1]-testx[0])-(testy[-1]-testy[-2])/(testx[-1]-testx[-2])
    # print(matrixa)
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
    x0=np.linspace(0,8,400)
    cubic_spline_out=np.zeros(x0.shape[0])
    for i in range(x0.shape[0]):
        cubic_spline_out[i]=cubic_spline(x0[i],testx,testy,M)
        
    return  cubic_spline_out
Sx=S(t,x)
Sy=S(t,y)
# print(Sy)
plt.scatter(np.linspace(0,8,400),Sx,s=1,label='x')
plt.scatter(np.linspace(0,8,400),Sy,s=1,label='y')
plt.scatter(t,x,s=10,label='x1')
plt.scatter(t,y,s=10,label='y1')
plt.legend()
plt.ylim(-2,2)
namefig='partb.png'
plt.savefig(script_directory/namefig,dpi=300)
plt.close()
#partc
fig,axs=plt.subplots(1,2,figsize=(10,8))
axs[0].scatter(x,y,s=10,label='interpolate point')
axs[0].scatter(Sx,Sy,s=1,label='interpolate function')
axs[0].legend()
t1=np.arange(1000)
phi1=np.arange(1000)*mt.pi/500
x1=(1-np.cos(phi1))*np.cos(phi1)
y1=(1-np.cos(phi1))*np.sin(phi1)
axs[1].scatter(x,y,s=10,label='interpolate point')
axs[1].scatter(x1,y1,s=1,label='reference function')

axs[1].legend()
namefig='partc.png'
plt.savefig(script_directory/namefig,dpi=300)
plt.close()