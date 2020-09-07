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

    start_date = datetime.datetime(start[0],start[1],start[2])
    end_date = datetime.datetime(end[0],end[1],end[2])
    df = pdr.get_data_yahoo(index, start=start_date, end=end_date)
    to_drop=["High","Low","Volume","Adj Close"]
    df.drop(to_drop, axis=1, inplace=True)

    df["weekday"] = df.index.weekday  # The day of the week with Monday=0, Sunday=6.
    # df["Close_5"] = df["Close"].rolling(window=5).mean().shift(1)
    # df["Close_30"] = df["Close"].rolling(window=30).mean().shift(1)
    # df["Close_150"] = df["Close"].rolling(window=150).mean().shift(1)

    # df["Close_150"] = df["Close"].ewm(span=150, adjust=False).mean().shift(1)

    df["em_150"] = df["Close"].ewm(span=150, adjust=False).mean().shift(1)
    df["Close_150"] =2*df["em_150"]- df["em_150"].ewm(com=0.5,adjust=False).mean().shift(1)

    df["mv_150"] = df["Close"].rolling(window=150).mean().shift(1)

    df = df[150:]
    df = df.loc[df["weekday"] == 0]
    df["Open next"] = df["Open"].shift(-1)
    df["quot"] = df["Open next"] / df["Open"]
    df=df[:-1]

    return df



def yield_gross(df,v):
    ## df["quot"] è il rapporto tra il prezzo open del primo giorno del mese successivo con
    ## il prezzo open del primo giorno del mese corrente
    ## v è il vettore che indica se sei o no nel mercato in quel mese
    prod=(v*df["quot"]+1-v).prod()
    n_weeks=len(v)/(12*4)
    return (prod-1)*100,((prod**(1/n_weeks))-1)*100


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


def yield_net(df, v,tax_cg=0.26,comm_bk=0.001):
    n_years = len(v)/(12*4)

    w, n = separate_ones(v)
    A = (w * np.array(df["quot"]) + (1 - w)).prod(axis=-1)  # A is the product of each group of ones of 1 for df["quot"]
    A1p = np.maximum(0, np.sign(A - 1))  # vector of ones where the corresponding element if  A  is > 1, other are 0
    Ap = A * A1p  # vector of elements of A > 1, other are 0
    Am = A - Ap  # vector of elements of A <= 1, other are 0
    An = Am + (Ap - A1p) * (1 - tax_cg) + A1p
    prod = An.prod() * ((1 - comm_bk) ** (2 * n))
    res=(prod - 1) * 100, ((prod ** (1 / n_years)) - 1) * 100
    return res


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