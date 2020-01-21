import pandas as pd
import numpy as np


def rolling_sr(stock_time_series,period):
    daily_r = (stock_time_series.shift(-1)/stock_time_series-1)
    er = daily_r.rolling(window=period,).mean()
    vol = daily_r.rolling(window=period,).std()
    res = er/vol
    res = res.shift(-period+1)
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
    ind_table = ind_table.apply(lambda x: x//(1/layer_num), axis=1)
    return ind_table

if __name__=="__main__":
    raw_data = pd.read_csv("raw_data.csv", index_col=[0], header=[0, 1], parse_dates=True)
    raw_data = raw_data.dropna(axis=1)
    close_price = raw_data.loc[:,"Adj Close"]
    label_table = get_stock_performance(close_price,rolling_sr,5,period=100)
    print(label_table)





