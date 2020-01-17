from pandas_datareader import data as pdrd
import numpy as np
import pandas as pd
from datetime import datetime



stock_list = pd.read_csv("MSCI_US_IMI_HC_benchmark_info_js2.csv")
symbol = stock_list['Symbol'].tolist()


#data = pdrd.get_data_yahoo(symbol[:30],'01/01/2009',interval='d')
#data.to_csv("raw_data.csv")

raw_data = pd.read_csv("raw_data.csv",index_col=[0],header=[0,1],parse_dates=True)
raw_data = raw_data.dropna(axis=0)









