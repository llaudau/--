import numpy as np
import matplotlib.pyplot as plt
datax=np.array([0, 1 ,2 ,3, 4 ,5, 6 ,7 ,8 ,9])
datay=np.array([1.2, 2.8 ,4.5, 7.1, 9.8, 13.5, 17.2, 21.9, 27.1, 33.2])
datasigma=np.array([0.12, 0.24, 0.14, 0.33, 0.90, 0.11, 0.45, 0.43 ,0.59 ,0.10])
from pathlib import Path
script_directory = Path(__file__).parent
# copy of ass1 gauss_method solve linear equations
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
    return reverse(a)

def fit_n_d(x,y,sig,n):
    dim=n+1
    def U(i,j,x):
        return np.sum(x**(i+j)/sig**2)
    def v(i,x,y):
        return np.sum(y*x**(i)/sig**2)
    A=np.zeros((dim,dim+1))
    for i in range(dim):
        for j in range(dim+1):
            if j==dim:
                A[i,j]=v(i,x,y)
            else:
                A[i,j]=U(i,j,x)
    return gauss_method(A)

def poly(x,para):
    order=0
    out=0
    for i in para:
        out+=i*x**order
        order+=1
    return out
    
x=np.linspace(0,10,500)
plt.errorbar(datax,datay,datasigma,fmt='',ecolor='black',elinewidth=1,capsize=2)
for i in range(3):
    parai=fit_n_d(datax,datay,datasigma,i+1)
    print(f'ord {i+1} : ',parai)
    plt.scatter(x,poly(x,parai),s=0.5,label=f'ord={i+1}')
plt.legend()
name='parta'
plt.savefig(script_directory/name,dpi=300)