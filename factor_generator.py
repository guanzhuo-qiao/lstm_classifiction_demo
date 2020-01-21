import pandas as pd
import numpy as np

def wr(high,close,low,period=(5,14,20)):
    h1 = high.rolling(window=period[0]).max()
    h2 = high.rolling(window=period[1]).max()
    l = low.rolling(window=period[2]).min()
    res = (h1-close)/(h2-l)
    res.columns=["wr"]
    return res

def clv(high,close,low):
    res = ((close-low)-(high-close))/(high-low)
    res.columns = ["clv"]
    return res

def ad(high,close,low,period=14):
    clv_data = clv(high,close,low)
    res = clv_data.rolling(window=period).sum()
    res.columns = ["ad"]
    return res

def ppo(close,period=(26,12)):
    fast_ema = close.ewm(span=period[1]).mean()
    slow_ema = close.ewm(span=period[0]).mean()
    res = 100*(fast_ema-slow_ema)/fast_ema
    res.columns = ["ppo"]
    return res

def pvo(volume,period=(26,12)):
    res = ppo(volume,period)
    res.columns = ["pvo"]
    return res

def so(high,close,low,period=(39,3)):
    h = high.rolling(window=period[0]).max()
    l = low.rolling(window=period[0]).min()
    k = (close-l)/(h-l)
    d = k.rolling(window=period[1]).mean()
    j = d.rolling(window=period[1]).mean()
    res = pd.concat((k,d,j),axis=1)
    res.columns=["k","d","j"]
    return res

def macd(close,period=(12,26,9)):
    diff = close.ewm(span=period[0]).mean()-close.ewm(span=period[1]).mean()
    dea = diff.ewm(span=period[2]).mean()
    res = diff-dea
    res.columns = ["macd"]
    return res

def bb(close,period=(5,14,20)):
    mid = close.rolling(window=period[0]).mean()
    sd = close.rolling(window=period[1]).std()
    res = (close-mid)/sd
    res.columns = ["bb"]
    return res

def cmf(high,close,low,volume,period=20):
    ad=((close-low)-(high-close))/(high-low)*volume
    res = ad.rolling(window=period).sum()/volume.rolling(window=period).sum()
    res.columns = ["cmf"]
    return res

def rsi(close,period=14):
    diff = close-close.shift(1)
    ma1 = diff.applymap(lambda x: max(x,0))
    ma1 = ma1.rolling(window=period).mean()
    ma2 = diff.applymap(abs)
    ma2 = ma2.rolling(window=period).mean()
    res = ma1/ma2*100
    res.columns = ["rsi"]
    return res


def get_tech_factors(dt: "DataFrame") -> "DataFrame":
    """
    dt: pandas dataframe of one stock
    return: pandas dataframe of the factors
    """
    # Williams R
    wr_list = wr(dt["High"],dt["Adj Close"],dt["Low"])
    clv_list = clv(dt["High"],dt["Adj Close"],dt["Low"])
    ad_list = ad(dt["High"],dt["Adj Close"],dt["Low"])
    ppo_list = ppo(dt["Adj Close"])
    pvo_list = pvo(dt["Volume"])
    so_list = so(dt["High"],dt["Adj Close"],dt["Low"])
    macd_list = macd(dt["Adj Close"])
    bb_list = bb(dt["Adj Close"])
    cmf_list = cmf(dt["High"],dt["Adj Close"],dt["Low"],dt["Volume"])
    rsi_list =rsi(dt["Adj Close"])
    res = pd.concat([wr_list,ppo_list,
                     pvo_list,so_list,macd_list,bb_list,
                     rsi_list],axis=1)
    return res

def get_x_data(dt):
    table = pd.DataFrame()
    for symbols in dt.loc[:,"Adj Close"].columns:
        stock_data = dt.loc[:,(slice(None),symbols)]
        factor_data = get_tech_factors(stock_data)
        factor_data.columns = pd.MultiIndex.from_product([factor_data.columns.to_list(),[symbols]])
        table = pd.concat([table,factor_data],axis=1)
        print(symbols)
    return table


if __name__=="__main__":
    raw_data = pd.read_csv("raw_data.csv", index_col=[0], header=[0, 1], parse_dates=True)
    raw_data = raw_data.dropna(axis=1)
    #table = get_tech_factors(raw_data.loc[:,(slice(None),"ABT")])
    x_table = get_x_data(raw_data)