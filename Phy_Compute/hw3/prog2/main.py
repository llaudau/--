import numpy as np
from scipy.special import roots_legendre
from scipy.integrate import quad
import math as mt
import matplotlib.pyplot as plt
import os
from pathlib import Path
script_directory = Path(__file__).parent

def f(x):
    return np.exp(-x)/x

def trapezoid(g,a,b,N):
    sum=0
    x=np.linspace(a,b,N)
    for i in range(N):
        if i==0 or i==N-1:
            sum+=g(x[i])*1/2
        else:
            sum+=g(x[i])
    return sum*(b-a)/N

def simpson(g,a,b,N):
    sum=0
    x=np.linspace(a,b,N)
    if (N-1)%2==1:
        for i in range(N-3):
            if i==0 or i==N-4:
                sum+=g(x[i])*1/3
            elif i%2==0:
                sum+=g(x[i])*2/3
            else:
                sum+=g(x[i])*4/3
        sum+=3/8*(g(x[N-4]+3*g(x[N-3])+3*g(x[N-2])+g(x[N-1])))
    
    return sum*(b-a)/N

def gauss_legendre(g,a,b,N):
    x_gauss, w = roots_legendre(N)
    x = (b-a)/2 * x_gauss + (b+a)/2
    y=g(x)
    weighted_sum = np.sum(w*y)
    result = (b-a)/2 * weighted_sum
    return result
a,b=1,100
N=[10,100,1000]
for i in N:
    print(f"trapezoid, invervals={i} : ")
    print(trapezoid(f,a,b,i))
    print(f"simpson, invervals={i} : ")
    print(simpson(f,a,b,i))
    print(f"GL, interval={i} : ")
    print(gauss_legendre(f,a,b,i))