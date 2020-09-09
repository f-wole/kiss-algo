from utils import yield_net,yield_gross,get_ins,get_outs,drawdown,add_max_loss,select_first_one
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize


data_path=sys.argv[1]
params_path=sys.argv[2]
out_path=sys.argv[3]
fast=int(sys.argv[4])
slow=int(sys.argv[5])
max_loss=sys.argv[6]

print("### Starting evaluation of ",out_path)

if not out_path.endswith("/"):
    out_path+="/"
if not data_path.endswith("/"):
    data_path+="/"

with open(data_path+"data.pkl","rb") as r:
    data=pickle.load(r)
with open(params_path,"rb") as r:
    a,b=pickle.load(r)

if max_loss in ["NO","no","No"]:
    max_loss_bool=False
else:
    max_loss_bool=True
    max_loss=float(max_loss)

# modified KISS
v_=np.array(data["Close_fast"]>data["Close_slow"],dtype=int)
if max_loss_bool:
    loss=select_first_one(np.array(data["delta2"]<=max_loss))
    v=add_max_loss(v_,loss)
else:
    v=v_

v_bh=np.ones(v.shape[0])

v_tune_ = np.array(data["Close_fast"] > data["Close_slow"] * a + b)
if max_loss_bool:
    loss=select_first_one(np.array(data["delta2"]<max_loss))
    v_tune=add_max_loss(v_tune_,loss)
else:
    v_tune=v_tune_


res={}
res["type"]=["Buy & hold","Simple Kiss","Tuned Kiss"]
res["Total Gross Yield"]=[yield_gross(data,v_bh)[0],yield_gross(data,v)[0],yield_gross(data,v_tune)[0]]
res["Total Net Yield"]=[yield_net(data,v_bh)[0],yield_net(data,v)[0],yield_net(data,v_tune)[0]]
res["Annual Gross Yield"]=[yield_gross(data,v_bh)[1],yield_gross(data,v)[1],yield_gross(data,v_tune)[1]]
res["Annual Net Yield"]=[yield_net(data,v_bh)[1],yield_net(data,v)[1],yield_net(data,v_tune)[1]]
res["Max Drawdown %"]=[drawdown(data,v_bh),drawdown(data,v),drawdown(data,v_tune)]
df=pd.DataFrame(res)
df.to_excel(out_path+"results.xlsx",index=False)

npdata=np.array(data["Open"])
npdataclose=np.array(data["Close"])
npmeanfast=np.array(data["Close_fast"])
npmeanslow=np.array(data["Close_slow"])
plt.close()
plt.figure(figsize=(35,10))

plt.plot(npdata, label="Open")
# plt.plot(npdataclose, label="Close")
plt.plot(npmeanfast,label="mean "+str(fast))
plt.plot(npmeanslow,label="mean "+str(slow))
plt.plot(get_ins(npdata,v)[0],get_ins(npdata,v)[1],'^', markersize=10, color='g',label="default KISS in")
plt.plot(get_outs(npdata,v)[0],get_outs(npdata,v)[1],'v', markersize=10, color='r',label="default KISS out")
plt.legend(fontsize=20)
plt.grid(axis="both")
plt.title("data_simple",fontsize=25)
plt.xticks(np.arange(v.shape[0])[::10],data.index.strftime("%d/%m/%Y")[::10],rotation=90)
plt.savefig(out_path+"plot_simple")

######
plt.close()
plt.figure(figsize=(35,10))
plt.plot(npdata, label="Open")
# plt.plot(npdataclose, label="Close")
plt.plot(npmeanfast,label="mean "+str(fast))
plt.plot(npmeanslow,label="mean "+str(slow))
plt.plot(get_ins(npdata,v_tune)[0],get_ins(npdata,v_tune)[1],'^', markersize=10, color='b',label="tuned KISS in")
plt.plot(get_outs(npdata,v_tune)[0],get_outs(npdata,v_tune)[1],'v', markersize=10, color='y',label="tuned KISS out")
plt.legend(fontsize=20)
plt.grid(axis="both")
plt.title("data_tuned",fontsize=25)
plt.xticks(np.arange(v.shape[0])[::10],data.index.strftime("%d/%m/%Y")[::10],rotation=90)
plt.savefig(out_path+"plot_tuned")
