# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit
from glob import glob
import datetime
import os

path='/Users/jason/Dropbox/AWS/GC-Net_L0_analysis/'
os.chdir(path)

base_dir = './input_data/raw/'
files = sorted(glob(base_dir+'*'))

n_stations=len(files)

fn='./ancillary/StationDescription ca.2000 GCN w NGRIP.csv'
info_all = pd.read_csv(fn)
info_all.columns

fn='./ancillary/varnames_all.txt'
info=pd.read_csv(fn, header=None, delim_whitespace=True)
info.columns=['id','varnam']
print(info.columns)
print(info.varnam)

varsx=info.varnam[3:47]
varsx=info.varnam[3:29]
n_vars=len(varsx)

th=1 ; th=2 # line thickness
formatx='{x:,.3f}'; fs=24 ; fs=18
plt.rcParams["font.size"] = fs
plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['axes.edgecolor'] = 'k'
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.8
plt.rcParams['grid.color'] = "#cccccc"
plt.rcParams["legend.facecolor"] ='w'
plt.rcParams["mathtext.default"]='regular'
plt.rcParams['grid.linewidth'] = th/2
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['figure.figsize'] = 12, 20

batch_name=['a','Tsnow','b']
lo=[5,21,30]
hi=[21,31,40]

# asa
for k in range(0,1):
    for i,fn in enumerate(files): # all
    # for i,fn in enumerate(files[10:11]): # SDM
    # for i,fn in enumerate(files[11:12]): # NSE
    # for i,fn in enumerate(files[12:13]): # NGRIP
    # for i,fn in enumerate(files[12:]): # all

        stnum=fn.split('/')[-1][0:2]
        v=np.where(info_all["Station Number"]==int(stnum)) ; v=v[0][0]
        print(v,stnum,info_all["Station Number"][v],info_all.nickname[v],fn)
        df=pd.read_csv('./input_data/'+info_all.nickname[v]+'.csv')
        # print(df.columns)
        varsx=df.columns[lo[k]:hi[k]]
        n_vars=len(varsx)
        df['time'] = pd.to_datetime(df.date)
        # df.P[df.P<600]=np.nan
        # df.P[df.P<600]=np.nan
        
        plt.close()
        fig, ax = plt.subplots(n_vars,1)
    
        cc=0
        ax[0].set_title(info_all.name[v]+' '+str(df.date[0])[0:10]+' to '+str(df.date[len(df)-1])[0:10])
    
        for j in range(0,n_vars):
            # ax[cc].set_title(varsx[cc])
            if cc<n_vars-1: ax[cc].get_xaxis().set_visible(False)
            # ax[cc].legend(prop={'size': fs*mult})
            # ax[cc].set_ylabel(varsx[cc])
    
            temp=df[varsx[cc]]
            temp[temp>990]=np.nan
            ax[cc].plot(df['time'],temp,label=varsx[cc])
            ax[cc].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
            cc+=1
    
        ly='p'
        if ly=='x':plt.show()
        # plt.show()
        figpath='./figs/'
        if ly=='p':
            fig.savefig(figpath+info_all.nickname[v]+'_'+batch_name[k]+'_v0.png',bbox_inches='tight', dpi=200)


