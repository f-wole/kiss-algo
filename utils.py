import numpy as np
import datetime
import pickle
import pandas as pd
import pandas_datareader as pdr
from keras.models import Sequential
from keras.optimizers import RMSprop,Adam
from keras.layers import Dense,Dropout,BatchNormalization,Conv1D,Flatten,MaxPooling1D,LSTM
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

def get_data_yahoo(start,end,index='^GSPC'):

    # input format: (year, day, month)

    #
    # returns indices and moving averages for all months between start (included) and end (excluded)
    # Tested

    start[0] -= 2
    start_date = datetime.datetime(start[0],start[1],start[2])
    end_date = datetime.datetime(end[0],end[1],end[2])
    df = pdr.get_data_yahoo(index, start=start_date, end=end_date)
    df.drop("Adj Close", axis=1, inplace=True)

    first_days = []
    # First year
    for month in range(start[1], 13):
        first_days.append(min(df[str(start[0]) + "-" + str(month)].index))
    # Other years
    for year in range(start[0] + 1, end[0]):
        for month in range(1, 13):
            first_days.append(min(df[str(year) + "-" + str(month)].index))
    # Last year
    for month in range(1, end[1] + 1):
        first_days.append(min(df[str(end[0]) + "-" + str(month)].index))

    dfm = df.resample("M").mean()
    dfm = dfm[:-1]  # As we said, we do not consider the month of end_date

    dfm["fd_cm"] = first_days[:-1]
    dfm["fd_nm"] = first_days[1:]
    dfm["fd_cm_close"] = np.array(df.loc[first_days[:-1], "Close"])
    dfm["fd_nm_close"] = np.array(df.loc[first_days[1:], "Close"])
    dfm["ratio"] = dfm["fd_nm_close"].divide(dfm["fd_cm_close"])

    dfm["mv_avg_3"] = dfm["Close"].rolling(window=3).mean().shift(1)
    dfm["mv_avg_6"] = dfm["Close"].rolling(window=6).mean().shift(1)
    dfm["mv_avg_12"] = dfm["Close"].rolling(window=12).mean().shift(1)
    dfm["mv_avg_24"] = dfm["Close"].rolling(window=24).mean().shift(1)
    dfm["quot"] = dfm["fd_nm_close"].divide(dfm["fd_cm_close"])

    dfm = dfm.iloc[24:, :]  # we remove the first 24 months, since they do not have the 2-year moving average

    indexes=["High","Low","Open","Close","Volume"]
    for index in indexes:
        dfm[index+"_avg"]=dfm[index]
        dfm=dfm.drop(index,axis=1)

    return dfm



def yield_gross(df,v):
    ## df["quot"] è il rapporto tra il prezzo open del primo giorno del mese successivo con
    ## il prezzo open del primo giorno del mese corrente
    ## v è il vettore che indica se sei o no nel mercato in quel mese
    prod=(v*df["quot"]+1-v).prod()
    n_years=len(v)/12
    return (prod-1)*100,((prod**(1/n_years))-1)*100


def separate_ones(u):
    u_ = np.r_[0, u, 0]
    i = np.flatnonzero(u_[:-1] != u_[1:])
    v, w = i[::2], i[1::2]
    if len(v) == 0:
        return np.zeros(len(u)), 0

    n, m = len(v), len(u)
    o = np.zeros(n * m, dtype=int)

    r = np.arange(n) * m
    o[v + r] = 1

    if w[-1] == m:
        o[w[:-1] + r[:-1]] = -1
    else:
        o[w + r] -= 1

    out = o.cumsum().reshape(n, -1)
    return out, n


def yield_net(df, v,tax_cg=0.26,comm_bk=0.01):
    n_years = len(v) / 12

    w, n = separate_ones(v)
    A = (w * np.array(df["quot"]) + (1 - w)).prod(axis=-1)  # A is the product of each group of ones of 1 for df["quot"]
    A1p = np.maximum(0, np.sign(A - 1))  # vector of ones where the corresponding element if  A  is > 1, other are 0
    Ap = A * A1p  # vector of elements of A > 1, other are 0
    Am = A - Ap  # vector of elements of A <= 1, other are 0
    An = Am + (Ap - A1p) * (1 - tax_cg) + A1p
    prod = An.prod() * ((1 - comm_bk) ** (2 * n))

    return (prod - 1) * 100, ((prod ** (1 / n_years)) - 1) * 100


def get_ins(npdata,v):
    x,y=[],[]
    for i in range(npdata.shape[0]):
        if v[i]==1:
            if i==0:
                x.append(i)
                y.append(npdata[i])
            else:
                if v[i-1]==0:
                    x.append(i)
                    y.append(npdata[i])
    # df=pd.DataFrame({"index":pd.DatetimeIndex(x),"value":np.array(y)})
    # df.index=pd.to_datetime(df.index)
    # df.set_index('index',inplace=True)
    return x,y

def get_outs(npdata,v):
    x, y = [], []
    for i in range(npdata.shape[0]):
        if v[i] == 0:
            if i == 0:
                x.append(i)
                y.append(npdata[i])
            else:
                if v[i - 1] == 1:
                    x.append(i)
                    y.append(npdata[i])
    # df=pd.DataFrame({"index":pd.DatetimeIndex(x),"value":np.array(y)})
    # df.index=pd.to_datetime(df.index)
    # df.set_index('index',inplace=True)
    return x, y