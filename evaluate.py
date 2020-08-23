from utils import yield_net,yield_gross
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize


data_path=sys.argv[1]
params_path=sys.argv[2]
out_path=sys.argv[3]

if not out_path.endswith("/"):
    out_path+="/"
if not data_path.endswith("/"):
    data_path+="/"

with open(data_path+"data.pkl","rb") as r:
    data=pickle.load(r)
with open(params_path,"rb") as r:
    a,b=pickle.load(r)

v=data["fd_cm_close"]>data["mv_avg_12"]
v_bh=np.ones(v.shape[0])

v_tune = data["fd_cm_close"] > data["mv_avg_12"] * a + b

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

plt.figure(figsize=(30,10))
plt.plot(data["fd_cm_close"], label="fd_cm_close")
plt.plot(data["mv_avg_12"], label="mv_avg_12")
plt.plot(v* max(data["fd_cm_close"]),label="default KISS in and out")
plt.plot(v_tune* max(data["fd_cm_close"]),label="TUNED in and out")
plt.legend(fontsize=20)
plt.grid(axis="both")
plt.title("data",fontsize=25)
plt.savefig(out_path+"plot")