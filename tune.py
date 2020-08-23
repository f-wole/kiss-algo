from utils import yield_net,yield_gross
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize


train_path=sys.argv[1]
out_path=sys.argv[2]

if not train_path.endswith("/"):
    train_path+="/"

with open(train_path+"data.pkl","rb") as r:
    train=pickle.load(r)

def tune(z):
    a,b=z
    v=train["fd_cm_close"]>train["mv_avg_12"]*a+b
    return -yield_net(train,v)[0]
rranges = (slice(-1, 2, 0.02), slice(-1, 2, 0.02))
resbrute = optimize.brute(tune, rranges, full_output=True, finish=optimize.fmin)
print("best parameters are : ", resbrute[0])
a,b=resbrute[0]

with open(out_path,"wb") as w:
    pickle.dump([a,b],w)
