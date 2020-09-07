from utils import yield_net,yield_gross,get_ins,get_outs
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize


data_path=sys.argv[1]
params_path=sys.argv[2]
out_path=sys.argv[3]

print("### Starting evaluation of ",out_path)

if not out_path.endswith("/"):
    out_path+="/"
if not data_path.endswith("/"):
    data_path+="/"

with open(data_path+"data.pkl","rb") as r:
    data=pickle.load(r)
with open(params_path,"rb") as r:
    a,b=pickle.load(r)

v=data["Open"]>data["Close_150"]
v_bh=np.ones(v.shape[0])

v_tune = data["Open"] > data["Close_150"] * a + b

with open(out_path+"results.txt","w",encoding="utf8") as w:
    print("Net yield on data with buy & hold: ",file=w)
    print("\t total = ",str(yield_net(data,v_bh)[0]),", annual = ",str(yield_net(data,v_bh)[1]),file=w)
    print("Gross yield on data with buy & hold: ",file=w)
    print("\t total = ",str(yield_gross(data,v_bh)[0]),", annual = ",str(yield_gross(data,v_bh)[1]),file=w)
    print("Net yield on data with default KISS: ",file=w)
    print("\t total = ",str(yield_net(data,v)[0]),", annual = ",str(yield_net(data,v)[1]),file=w)
    print("Gross yield on data with default KISS: ",file=w)
    print("\t total = ",str(yield_gross(data,v)[0]),", annual = ",str(yield_gross(data,v)[1]),file=w)
    print("Net yield on data with tuned KISS: ",file=w)
    print("\t total = ",str(yield_net(data,v_tune)[0]),", annual = ",str(yield_net(data,v_tune)[1]),file=w)
    print("Gross yield on data with tuned KISS: ",file=w)
    print("\t total = ",str(yield_gross(data,v_tune)[0]),", annual = ",str(yield_gross(data,v_tune)[1]),file=w)

npdata=np.array(data["Open"])
npmean=np.array(data["Close_150"])
npmean1=np.array(data["em_150"])
npmean2=np.array(data["mv_150"])
plt.close()
plt.figure(figsize=(35,10))

plt.plot(npdata, label="Open")
plt.plot(npmean,label="dema_150")
plt.plot(npmean1,label="em_150")
plt.plot(npmean2,label="mv_150")
plt.plot(get_ins(npdata,v)[0],get_ins(npdata,v)[1],'^', markersize=10, color='g',label="default KISS in")
plt.plot(get_outs(npdata,v)[0],get_outs(npdata,v)[1],'v', markersize=10, color='r',label="default KISS out")
plt.plot(get_ins(npdata,v_tune)[0],get_ins(npdata,v_tune)[1],'^', markersize=10, color='b',label="tuned KISS in")
plt.plot(get_outs(npdata,v_tune)[0],get_outs(npdata,v_tune)[1],'v', markersize=10, color='y',label="tuned KISS out")

plt.legend(fontsize=20)
plt.grid(axis="both")
plt.title("data",fontsize=25)
plt.xticks(np.arange(v.shape[0]),data.index,rotation=45)
plt.savefig(out_path+"plot")
