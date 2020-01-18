import pandas as pd
import numpy as np


def rolling_sr(stock_time_series,period):
    annual_r = (stock_time_series.shift(-period)/stock_time_series-1)/period*250
    vol = annual_r.rolling(window=period+1,).std().shift(-period)
    res = annual_r/vol
    return res

def get_stock_performance(dt: pd.DataFrame,func: "function",layer_num: int,*args,**kwargs) -> pd.DataFrame:
    """
    dt: 2d pd.DataFrame, rows: time; columns: stock symbols
    func: the function of computing the performance. like sharpe ratio. implemented on one stock time series
    layer_num: the number of layers that the stocks can be classified
    *args & **kwargs: parameters given to the func
    return: 2d pd.DataFrame, labels of the each stocks at each time
    """
    ind_table = dt.apply(func,axis=0,*args,**kwargs)
    ind_table = ind_table.rank(axis=1,pct=True,ascending=True)
    ind_table = ind_table.apply(lambda x: x//(1/layer_num)+1, axis=1)
    return ind_table







