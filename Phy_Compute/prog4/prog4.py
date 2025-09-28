import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math as mt

N=20 #from 1 to negative order
x0=-1
n=np.arange(N)
h=(1/10)**(n+1)
print('h=',h)
def f(x):
    return np.exp(5*x)

f1=(f(x0+h)-f(x0))/h
print('f\'=',f1)
print(mt.log(2))
print(1.00001-round(1.00001/mt.log(2))*mt.log(2))