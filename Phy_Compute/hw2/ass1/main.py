#in this task i use numpy package to better store matrixes and achieve matrix operation. 
import numpy as np

testA=[[6,-2,1,5],[1,5,2,17],[-1,1,4,13]]
testA=np.array(testA,float)

#standard output
def output(solve):
    index=1
    for i in solve:
        print(f'x_{index} = {i:.6f}')
        index+=1



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
    output(reverse(a))
    return



def jacobi_method(array):
#jacobi iteration method

    def seperate(array):
        rows=array.shape[0]
        D=np.diag(array)
        D_inverse=np.diag(1/D)
        D=np.diag(D)
        b=array[:,-1]
        B=array[:rows,:rows]-D
        return D_inverse,B,b

    def jacobi_iteration(D_inv,B,b):
        x0=np.zeros(D_inv.shape[0])
        # order=0
        while True:
            x1=D_inv@(b-B@x0)
            if np.sum(abs(x1-x0))/np.sum(abs(x0)+0.01)<=0.000001:
                break
            x0=x1
            # order+=1
            # print(x0)
            # if order==10:
            #     break
            
        return x0




    D_inv,B,b=seperate(array)
    out=jacobi_iteration(D_inv,B,b)
    output(out)
    return
print('gauss')
gauss_method(testA)
print("jacobi:")
jacobi_method(testA)