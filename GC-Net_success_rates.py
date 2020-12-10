# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import os

path='/Users/jason/Dropbox/AWS/GC-Net_L0_analysis/'
os.chdir(path)

th=1 ; th=2 # line thickness
formatx='{x:,.3f}'; fs=24 ; fs=16
plt.rcParams["font.size"] = fs
plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['axes.edgecolor'] = 'k'
plt.rcParams['axes.grid'] = False
# plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.8
plt.rcParams['grid.color'] = "#cccccc"
plt.rcParams["legend.facecolor"] ='w'
plt.rcParams["mathtext.default"]='regular'
plt.rcParams['grid.linewidth'] = th/2
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['figure.figsize'] = 12, 20

ancil_path='./ancillary/'
base_dir = './input_data/raw/'
files = sorted(glob(base_dir+'*.dat'))

n_stations=len(files)

fn=ancil_path+'StationDescription ca.2000 GCN.csv'
info_all = pd.read_csv(fn)
info_all.columns

fn=ancil_path+'varnames_all.txt'
info=pd.read_csv(fn, header=None, delim_whitespace=True)
info.columns=['id','varnam']
print(info.columns)
print(info.varnam)

varsx=info.varnam[3:47]
varsx=info.varnam[3:29]
n_vars=len(varsx)

success_rate=np.zeros((n_stations,n_vars))
success_count=np.zeros((n_stations,n_vars))
counts=np.zeros((n_stations))
time0=['']*n_stations
time1=['']*n_stations

# for i,fn in enumerate(files[0:1]):
# for i,fn in enumerate(files[10:]):
for i,fn in enumerate(files):
    stnum=fn.split('/')[-1][0:2]
    v=np.where(info_all["Station Number"]==int(stnum))
    v=v[0][0]
    df=pd.read_csv(fn, header=None, delim_whitespace=True)
    df.columns=info.varnam
    n=len(df)
    counts[v]=n
    time0[v]=str(df["Year"][0])+' '+str(df["JulianDay"][0])
    time1[v]=str(df["Year"][n-1])+' '+str(df["JulianDay"][n-1])
    print(v,int(stnum),info_all.nickname[v],n)
    for j,var in enumerate(varsx):
        success=(n-len(df[var][df[var]>990]))/n
        success_rate[v,j]=success
        success_count[v,j]=n-len(df[var][df[var]>990])
        # print(var,str("%.2f"%success))

 

#%%
fig, ax = plt.subplots(1,1)
from mpl_toolkits.axes_grid1 import make_axes_locatable

img = ax.imshow(success_rate*100)

# -------------------------- x axis
x_label_list = varsx
ax.set_xticks(np.arange(0,n_vars, 1.0))
ax.set_xticklabels(x_label_list,rotation=90)

# -------------------------- y axis
y_label_list = info_all.nickname
ax.set_yticks(np.arange(0,n_stations, 1.0))
ax.set_yticklabels(y_label_list)

ax.set_title('GC-Net success rate in L0 N. Bayou share')

# -------------------------- colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.06)
plt.colorbar(img, cax=cax)

plt.title('%',size=fs*0.9)

n_years_array=np.zeros(n_stations)
success_array=np.zeros(n_stations)


for i,fn in enumerate(files):
    stnum=fn.split('/')[-1][0:2]
    v=np.where(info_all["Station Number"]==int(stnum)) ; v=v[0][0]
    ny=np.sum(counts[v])/8760
    max_poss=n_vars*np.sum(counts[v])
    success_summary=np.sum(success_count[v,:])/max_poss
    print(info_all.nickname[v],str("%.1f"%ny),str("%.2f"%success_summary))
    n_years_array[i]=ny
    success_array[i]=success_summary
    
s=np.flip(np.argsort(success_array))
print('success ')
for ii,fn in enumerate(files):
    i=s[ii]
    print(info_all.nickname[i],str("%.2f"%success_array[i]),time0[i],time1[i])

s=np.flip(np.argsort(n_years_array))
print('n years')
for ii,fn in enumerate(files):
    i=s[ii]
    print(info_all.nickname[ii],str("%.1f"%n_years_array[i]),time0[i],time1[i])

plt.show()
 
wo_fig=1
figpath='./figs/'
if wo_fig:fig.savefig(figpath+'GC-Net_L0_success_rates.png',bbox_inches='tight', dpi=200)
