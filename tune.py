from utils import yield_net,yield_gross,add_max_loss
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize


train_path=sys.argv[1]
out_path=sys.argv[2]
max_loss=sys.argv[3]

print("### Starting tuning of parameters ")

if not train_path.endswith("/"):
    train_path+="/"

with open(train_path+"data.pkl","rb") as r:
    train=pickle.load(r)

if max_loss in ["NO","no","No"]:
    max_loss_bool=False
else:
    max_loss_bool=True
    max_loss=float(max_loss)


def tune(z):
    a,b=z
    v_ = np.array(train["Close_fast"] > train["Close_slow"] * a + b,dtype=int)
    if max_loss_bool:
        loss = np.array(train["delta2"] < max_loss)
        v = add_max_loss(v_, loss)
    else:
        v=v_
    return -yield_net(train,v)[0]
rranges = (slice(-1, 2, 0.1), slice(-1, 2, 0.1))
resbrute = optimize.brute(tune, rranges, full_output=True, finish=optimize.fmin)
# resbrute = optimize.minimize(tune,x0=np.array([1,1,1]), method='BFGS')

print("best parameters are : ", resbrute[0])
a,b=resbrute[0]

with open(out_path,"wb") as w:
    pickle.dump([a,b],w)
