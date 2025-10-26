# this file is used to get all data from akshare 
# and save as a npz file, a dictionary str::code -> data
# since now my target is the GEM stocks

import numpy as np
import pandas as pd
import akshare as ak
import time
import datetime
import os 
def get_code():
    script_dir=os.path.dirname(os.path.abspath(__file__))
    GEM_path = os.path.join(script_dir, "filter_stocks", "code_GEM.csv")
    code_GEM=pd.read_csv(GEM_path)['code']
    code_GEM_np=code_GEM.to_numpy(dtype=int)
    return code_GEM_np

def save_data(codes,days):
    
    dir={}
    #build numpy
    build=ak.stock_zh_a_hist(symbol=str(codes[0]), adjust='qfq').tail(days)
    col=build.columns.to_list()

    dates=build.iloc[:,0]

    for i in codes:
        try:
            df = ak.stock_zh_a_hist(symbol=str(i), adjust='qfq').tail(days)
            array=df.iloc[:,2:].to_numpy()
            dir[str(i)]=array
            print(f"Downloaded {i}")
            time.sleep(1)   # prevent being blocked
        except Exception as e:
            print(f"Failed for {i}: {e}")

    #save data
    script_dir=os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "filter_stocks", "GEM_data.npz")
    
    #save the name of column
    name_path=os.path.join(script_dir, "filter_stocks", "Readme.txt")
    l=['the corresponding 10 index of the 2nd dimension is: ']
    l=l+col[2:]
    with open(name_path,'w') as f:
        f.writelines(_+'\n' for _ in l)
        f.writelines(str(_)+',  \n' for _ in dates)
    # np.savez(data_path,**dir)
    return
save_data(get_code(),600)