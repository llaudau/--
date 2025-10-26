import numpy as np
import matplotlib.pyplot as plt
testx=np.array([0,0.3,0.6,0.9,1.2,1.5])
testy=np.tan(testx)
def part1(testx,testy):
    def generate(array,x,y,column,row):
        if column==0:
            array[row,column]=(x[0]-x[row])/(y[0]-y[row])
        else:
            array[row,column]=(x[column]-x[row])/(array[column,column-1]-array[row,column-1])
        return

    def iteration(array,x,y):
        length=x.shape[0]
        for column in range(length):
            for row in range(1,length):
                generate(array,x,y,column,row)

        return array
    array0=np.zeros((testx.shape[0],testx.shape[0]))
    iteration(array0,testx,testy)
    shit=[]
    for i in range(testx.shape[0]-1):
        shit.append(array0[i+1,i])
    return shit
shit=part1(testx,testy)
for i in shit:
    print(i)


#part2

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

def rational_fraction(x,testx,testy):
    array1=part1(testx,testy)
    result=0
    for i in range(testx.shape[0]-1):
        if i==0:
            result=(x-testx[-2])/(array1[-1])
        else:
            result=(x-testx[-2-i])/(array1[-1-i]+result)
    return result+testy[0]

Lag_res=np.zeros(500)
rf_res=np.zeros(500)
x=np.arange(500)*1.57/499
target_res=np.tan(x)

for i in range(500):
    Lag_res[i]=Lag(testx,testy,x[i])
    rf_res[i]=rational_fraction(x[i],testx,testy)

print('Lagrange:')
biaslag=np.sum((Lag_res-target_res)**2)
print(biaslag)
print('rational fraction:')
biasrf=np.sum((rf_res-target_res)**2)
print(biasrf)


# plt.scatter(x,Lag_res,label='lag',s=0.5)
# plt.scatter(x,target_res,label='tar',s=0.5)
# plt.scatter(x,rf_res,label='rf',s=0.5)
# plt.legend()
# plt.ylim(0,5)
# plt.show()