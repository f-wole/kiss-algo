import sys
import json
import pickle
import numpy as np
from utils import get_data_yahoo
import matplotlib.pyplot as plt

start=json.loads(sys.argv[1]) # format: (year, month) included
end=json.loads(sys.argv[2]) # format: (year, month) excluded
save_path=sys.argv[3]

if not save_path.endswith("/"):
    save_path+="/"

# FIX DATES
last_days={1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
dfm=get_data_yahoo([start[0],start[1],1],
                   [end[0],end[1],last_days[end[1]]])


with open(save_path+"data.pkl","wb") as w:
    pickle.dump(dfm,w)

dfm.to_excel(save_path+"data.xlsx")
dfu=dfm.index
dfu.drop_duplicates
dfu=list(dfu)
dates=["-".join(str(date).split("-")[:2]) for date in dfu]

with open(save_path+"dates.txt","w",encoding="utf8") as w:
    for date in dates:
        w.write(date+"\n")

plt.plot(dfm["fd_cm_close"])
plt.title("Close price of first day of current month")
plt.xticks(rotation=45)
plt.savefig(save_path+"plot")

print("### Successfully saved to ",save_path)
print("### From ",dates[0]," to ",dates[-1])
print("\t dfm shape = ",dfm.shape)