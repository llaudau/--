import numpy as np
import pandas as pd
import os
# impo

# this function is used to check and rank temperature of a series of data, especially stock data
# the passed array is ordered as below:
# code, days, highest price, lowest price, start value, end value

# now we define the temperature to be the commonsense of all player in the market, if they all 
# believe that the stock will go in certian direction just like a iron magnet
# we say the market has a low temperature. exp(-T)=abs(start-end)/abs(max-min)
# when T is big, we have chance to get profit
def check_T_per(array):
    shape=array.shape
    sum=0
    T_arr=(array[:,1]-array[:,2])/(array[:,2]-array[:,3])
    T_arr=abs(T_arr)
    return 1/(np.sum(T_arr)/shape[0])-1

array0=np.load('/Users/wangkehe/Git_repository/stock_strategy/filter/filter_stocks/GEM_data.npz')
codes=array0.files
for code in codes:
    data=array0[code]
    print(check_T_per(data))
array0.close()