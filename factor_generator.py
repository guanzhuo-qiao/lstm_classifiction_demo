import pandas as pd
import numpy as np



def get_tech_factors(dt: "DataFrame") -> "DataFrame":
    """
    dt: pandas dataframe of one stock
    return: pandas dataframe of the factors
    """
    # Williams R
    

if __name__=="__main__":
    raw_data = pd.read_csv("raw_data.csv", index_col=[0], header=[0, 1], parse_dates=True)
    raw_data = raw_data.dropna(axis=0)
