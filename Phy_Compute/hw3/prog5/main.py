import math as mt
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
script_directory = Path(__file__).parent

def meth1(f,g,x,y,a):
    order=1
    while True:
        gra_x,gra_y=g(x,y)
        newx,newy=(x-gra_x*a),(y-gra_y*a)
        if abs(f(newx,newy)-f(x,y))<1e-5:
            return f(newx,newy),newx,newy,order
        x=newx
        y=newy
        order+=1
        if order==30:
            print(" steepest gradient not 1e-5")
            return f(newx,newy),newx,newy,order

# we choose Fletcher–Reeves beta_n 

def meth2(f, g, x, y, a):
    order = 1
    gx, gy = g(x, y)
    px, py = -gx, -gy

    while True:
        # update x,y
        newx, newy = x + a * px, y + a * py

        # stop condition
        if abs(f(newx, newy) - f(x, y)) < 1e-5:
            return f(newx,newy), newx, newy, order

        # compute new gradient
        gx_new, gy_new = g(newx, newy)

        # Beta (Fletcher–Reeves)
        beta = (gx_new**2 + gy_new**2) / (gx**2 + gy**2)

        # update direction
        px, py = -gx_new + beta * px, -gy_new + beta * py

        # step
        x, y = newx, newy
        gx, gy = gx_new, gy_new

        order += 1
        if order > 15:
            print("not 1e-5")
            return f(newx,newy), newx, newy, order


def f(x,y):
    return (x**2+y-11)**2+(x+y**2-7)**2

def g(x,y):
    gradx=2*(x+y**2-7)+4*x*(x**2+y-11)
    grady=2*(x**2+y-11)+4*y*(x+y**2-7)
    return gradx,grady

print(meth1(f,g,3,-3,0.01))
print(meth2(f,g,3,-3,0.01))