import math as mt
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
script_directory = Path(__file__).parent


def meth1(f,a,b):
    o=0
    while True:
        mid=(a+b)/2
        if f(mid)==0 or abs(b-a)<1e-5:
            return mid,o
        if f(mid)*f(a)<0:
            b=mid
        else:
            a=mid
        print((a,b))
        o+=1

def meth2(f,g,x0):
    o=0
    while True:
        newx0=x0-f(x0)/g(x0)
        print(newx0)
        # if o==0:
        #     print("x1 : ",newx0)
        o+=1
        if abs(newx0-x0)<1e-5:
            return newx0,o
        x0=newx0
def meth3(f,x0,x1):
    newx0=x0-f(x1)*(x1-x0)/(f(x1)-f(x0))
    x0=x1
    print(x1)
    o=1
    while True:
        if abs(newx0-x0)<1e-5:
            return newx0,o
        x1=newx0
        newx0=newx0-f(newx0)*(newx0-x0)/(f(newx0)-f(x0))
        print(newx0)
        x0=x1
        o+=1

def f(x):
    return x-2*mt.sin(x)
def g(x):
    return 1-2*mt.cos(x)

def f1(x):
    return x**2-4*x*mt.sin(x)+(2*mt.sin(x))**2
def g1(x):
    return 2*x-4*mt.sin(x)-4*x*mt.cos(x)+8*mt.sin(x)*mt.cos(x)
# print("meth1")
# print(meth1(f,1.5,2))
# print("meth2")
# print(meth2(f,g,1.5))
# x1=2.076558200630435
# print("meth3")
# print(meth3(f,1.5,x1))

# print("meth1")
# print(meth1(f1,1.5,2))
print("meth2")
print(meth2(f1,g1,0.5))
x1=0.1961915594624118
print("meth3")
print(meth3(f1,0.5,x1))