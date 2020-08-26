from utils import yield_net,yield_gross
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize


train_path=sys.argv[1]
out_path=sys.argv[2]

print("### Starting tuning of parameters ")

if not train_path.endswith("/"):
    train_path+="/"

with open(train_path+"data.pkl","rb") as r:
    train=pickle.load(r)

def tune(z):
    a,b,c=z
    v=train["fd_cm_close"]>train["mv_avg_12"]*a+b*train["mv_avg_6"]+c*train["mv_avg_3"]
    return -yield_net(train,v)[0]
rranges = (slice(-1, 2, 0.1), slice(-1, 2, 0.1), slice(-1, 2, 0.1))
resbrute = optimize.brute(tune, rranges, full_output=True, finish=optimize.fmin)
# resbrute = optimize.minimize(tune,x0=np.array([1,1,1]), method='BFGS')

print("best parameters are : ", resbrute[0])
a,b,c=resbrute[0]

with open(out_path,"wb") as w:
    pickle.dump([a,b,c],w)
