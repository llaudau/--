import math as mt
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
script_directory = Path(__file__).parent


def simpson(g,k,a,b,N):
    sum=0
    x=np.linspace(a,b,N)
    if (N-1)%2==1:
        for i in range(N-3):
            if i==0 or i==N-4:
                sum+=g(k,x[i])*1/3
            elif i%2==0:
                sum+=g(k,x[i])*2/3
            else:
                sum+=g(k,x[i])*4/3
        sum+=3/8*(g(k,x[N-4]+3*g(k,x[N-3])+3*g(k,x[N-2])+g(k,x[N-1])))
    
    return sum*(b-a)/N

def simpson1(g,k,a,b,N):
    sum=0
    x=np.linspace(a,b,N+1)
    for i in range(N//2):
        sum+=1/3*g(k,x[2*i])+4/3*g(k,x[2*i+1])+1/3*g(k,x[2*i+2])
    return sum*(b-a)/N

def f(k,x):
    return np.exp(x)*np.cos(k*x)

K=[100000,1000000]#1000,10000,100000,1000000
alpha=[0.01,0.1,1,10,100]#


# for k in K:
#     for alpha1 in alpha:
#         n=int(k*alpha1)
#         result=simpson1(f,k,0,1,n)
#         print(f"k = {k}, n = {n}")
#         print(result)


#part c

#interpolate
def interpolate_parameter(x,y):
    A = np.vander(x, 3)
    coefficients = np.linalg.solve(A, y)
    return coefficients

def intepolate_integral(k,n):
    sum=0
    h=1/(2*n)
    for i in range(n):
        x0=(2*i+1)/(2*n)
        # print(x0)
        x=np.array([(2*i)/(2*n),(2*i+1)/(2*n),(2*i+2)/(2*n)])
        y=np.exp(x)
        c,b,a=interpolate_parameter(x,y)
        C=c
        B=b+2*c
        A=a+x0**2*c+b*x0
        # print(c,b,a)
        # print(2/(k**3)*(mt.sin(h*k)*((-2*C+A*k**2+C*h**2*k**2)*mt.cos(k*x0)-B*k *mt.sin(k*x0))+h*k*mt.cos(k*h)*(2*C*mt.cos(k*x0)+B*k*mt.sin(k*x0))))
        sum+=2/(k**3)*(mt.sin(h*k)*((-2*C+A*k**2+C*h**2*k**2)*mt.cos(k*x0)-B*k *mt.sin(k*x0))+h*k*mt.cos(k*h)*(2*C*mt.cos(k*x0)+B*k*mt.sin(k*x0)))
    return sum
# print(intepolate_integral(1000000,1000000))
# a,b,c,x0,k,h=1.0005598411081316, 0.9880069540901326 ,0.5810381558502941,0.15,1,0.05
# shit=2/(k**3)*(mt.sin(h*k)*((-2*c+a*k**2+c*h**2*k**2)*mt.cos(k*x0)-b*k *mt.sin(k*x0))+h*k*mt.cos(k*h)*(2*c*mt.cos(k*x0)+b*k*mt.sin(k*x0)))
# print(shit)
for k in K:
    for alpha1 in alpha:
        n=int(k*alpha1)
        result=intepolate_integral(k,n)
        print(f"k = {k}, n = {n}")
        print(result)
