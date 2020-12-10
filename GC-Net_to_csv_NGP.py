# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import pandas as pd
from glob import glob
import datetime
import os
import warnings
warnings.filterwarnings("ignore")

path='/Users/jason/Dropbox/AWS/GC-Net_L0_analysis/'
os.chdir(path)

base_dir = './input_data/raw/'
files = sorted(glob(base_dir+'*.dat'))
files=['./input_data/raw/14c.dat']

n_stations=len(files)

fn='./ancillary/StationDescriptionNGRIP.csv'
info_all = pd.read_csv(fn)
info_all.columns

fn='./ancillary/varnames_gdf.txt'
info=pd.read_csv(fn, header=None, delim_whitespace=True)
info.columns=['id','varnam']
print(info.columns)
print(info.varnam)

varsx=info.varnam[3:47]
varsx=info.varnam[3:29]
n_vars=len(varsx)


missings=np.zeros((n_stations,n_vars))
missing_count=np.zeros((n_stations,n_vars))
counts=np.zeros((n_stations))

for i,fn in enumerate(files): # all sations
    stnum=fn.split('/')[-1][0:2]
    v=np.where(info_all["Station Number"]==int(stnum)) ; v=v[0][0]
    print(v,stnum,info_all["Station Number"][v],info_all.nickname[v],fn)
    # print(stnum)
    # sasdds
    df=pd.read_csv(fn, header=None, delim_whitespace=True)
    n=len(df)
    df.columns=info.varnam
    df=df.drop(columns=['WindSpeed@2m','WindSpeed@10m','q1','q2','q3','q4','Albedo'])
    df["date"]=np.nan 
    du=1
    if du:
        for j in range(0,n):
        # for j in range(0,25):
            doy=int(df["JulianDay"][j])
            hr=round((df["JulianDay"][j]-doy)*24)
            if hr<24:
                hour=str(hr).zfill(2)
                doy=str(int(df["JulianDay"][j])).zfill(3)
                print(stnum,df["Year"][j],df["JulianDay"][j],hour,doy)
                # xxx
                strx=str(df["Year"][j])+str(doy)+str(hour)
                # print(strx)
                df["date"][j]=datetime.datetime.strptime(strx, '%Y%j%H')
        # print(stnum,df["Year"][j],df["JulianDay"][j],df["date"][j],hour,strx)
        
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    if du:df.to_csv('./input_data/'+info_all.nickname[v]+'.csv')
# #%%
# fig, ax = plt.subplots(1,1)
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# img = ax.imshow(missings*100)

# # -------------------------- x axis
# x_label_list = varsx
# ax.set_xticks(np.arange(0,n_vars, 1.0))
# ax.set_xticklabels(x_label_list,rotation=90)

# # -------------------------- y axis
# y_label_list = info_all.nickname
# ax.set_yticks(np.arange(0,n_stations, 1.0))
# ax.set_yticklabels(y_label_list)

# ax.set_title('GC-Net flagged missing data in L0 share')

# # -------------------------- colorbar
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(img, cax=cax)

# plt.title('% missing')

# n_years_array=np.zeros(n_stations)
# missing_array=np.zeros(n_stations)

# for i,fn in enumerate(files):
#     stnum=fn.split('/')[-1][0:2]
#     v=np.where(info_all["Station Number"]==int(stnum))
#     ny=np.sum(counts[v[0]])/8760
#     max_poss=n_vars*np.sum(counts[v[0]])
#     missing_summary=np.sum(missing_count[v[0],:])/max_poss
#     print(info_all.nickname[v[0][0]],str("%.1f"%ny),str("%.2f"%missing_summary))
#     n_years_array[i]=ny
#     missing_array[i]=missing_summary
    
# s=np.flip(np.argsort(missing_array))
# print('missing ')
# for ii,fn in enumerate(files):
#     i=s[ii]
#     print(info_all.nickname[i],str("%.2f"%missing_array[i]))

# s=np.flip(np.argsort(n_years_array))
# print('n years')
# for ii,fn in enumerate(files):
#     i=s[ii]
#     print(info_all.nickname[ii],str("%.1f"%n_years_array[i]))

    # print(len(df))
# plt.plot(df.SnowHeight1)
# plt.plot(df.SnowHeight2)
# th=1 ; th=2
# formatx='{x:,.3f}'; fs=24 ; fs=18
# plt.rcParams["font.size"] = fs
# plt.rcParams['axes.facecolor'] = 'w'
# plt.rcParams['axes.edgecolor'] = 'k'
# plt.rcParams['axes.grid'] = False
# plt.rcParams['axes.grid'] = True
# plt.rcParams['grid.alpha'] = 0.8
# plt.rcParams['grid.color'] = "#cccccc"
# plt.rcParams["legend.facecolor"] ='w'
# plt.rcParams["mathtext.default"]='regular'
# plt.rcParams['grid.linewidth'] = th/2
# plt.rcParams['axes.linewidth'] = 1
# plt.rcParams['figure.figsize'] = 12, 12.3


# PROMICE_temperature='AirTemperatureHygroClip(C)' ; PROMICE_temperature_name='HygroClip'
# PROMICE_temperature='AirTemperature(C)' ; PROMICE_temperature_name='PT100'

# du_q=1

# if du_q:
#     print(df_interpol.columns)
#     # sss
#     # q1 = RH_ice2water(df_samira['RelativeHumidity'] ,
#     #                                            df_samira['AirTemperatureC'])
#     # df_samira['SpecHum_ucalg'] = RH2SpecHum(df_samira['RelativeHumidity'] ,
#     #                                                df_samira['AirTemperatureC'] ,
#     #                                                df_samira['AirPressurehPa'] )*1000
#     ytit='air temperature difference, °C'
#     xtit='shortwave downward, W m$^{-2}$'
#     ytit='PROMICE specific humidity, g/kg'
#     xtit='GC-Net specific humidity, g/kg'
    
#     y=df_interpol['SpecHum']
#     x1=df_interpol['SpecHum']/df_interpol['SpecHum1']
#     x2=df_interpol['SpecHum']/df_interpol['SpecHum2']
#     fig, ax = plt.subplots(figsize=(10,10))
#     color='b'
#     plt.scatter(x1,y,marker='s',color=color,label= 'GC-Net 1')
#     x=x1
#     v=np.where((np.isfinite(x))&(np.isfinite(y)))
#     b, m = polyfit(x[v[0]], y[v[0]], 1)
#     xx=[np.min(x[v[0]]),np.max(x[v[0]])]
#     yy=[b + m * xx[0],b + m * xx[1]]
#     plt.plot(xx, yy, '--',c=color)
#     print("range ",yy[1]-yy[0])
#     print("range min ",yy[1])
#     print("range max ",yy[0])

#     k=0
#     RMSD,R,bias,N =gnl.comp_stats(df_interpol[varname1[k]],df_interpol[PROMICE_temperature])
#     # print("RMSD",varname1[k],PROMICE_temperature,RMSD)
#     # print("R",varname1[k],PROMICE_temperature,R)
#     # print("bias",varname1[k],PROMICE_temperature,bias)
#     # print("N",varname1[k],PROMICE_temperature,N)    
    
#     cc=0
#     mult=1
#     xx0=1.01 ; yy0=0.99 ; dy=-0.15
#     props = dict(boxstyle='round', facecolor='w', alpha=0.5,edgecolor='w')
#     ax.text(xx0,yy0, 'N: '+str('%.0f'%N)+'\n'
#             +'correlation: '+str('%.3f'%R)+'\n'
#             +'bias: '+str('%.2f'%bias)+'\n'
#             +'RMSD: '+str('%.2f'%RMSD)+'\n'
#             ,transform=ax.transAxes, fontsize=fs*mult,
#             verticalalignment='top', bbox=props,rotation=0,color=color, rotation_mode="anchor")  

#     color='r'    
#     x=x2
#     plt.scatter(x,y,marker='s',color=color,label= 'GC-Net 2')
#     v=np.where((np.isfinite(x))&(np.isfinite(y)))
#     b, m = polyfit(x[v[0]], y[v[0]], 1)
#     yy=[b + m * xx[0],b + m * xx[1]]
#     plt.plot(xx, yy, '--',c=color)

#     k=1
#     RMSD,R,bias,N =gnl.comp_stats(df_interpol[varname1[k]],df_interpol[PROMICE_temperature])
#     print("RMSD",varname1[k],PROMICE_temperature,RMSD)
#     print("R",varname1[k],PROMICE_temperature,R)
#     print("bias",varname1[k],PROMICE_temperature,bias)
#     print("N",varname1[k],PROMICE_temperature,N)    
    
#     cc+=1
    
#     yy0+=dy*cc
#     ax.text(xx0,yy0, 'N: '+str('%.0f'%N)+'\n'
#             +'correlation: '+str('%.3f'%R)+'\n'
#             +'bias: '+str('%.2f'%bias)+'\n'
#             +'RMSD: '+str('%.2f'%RMSD)+'\n'
#             ,transform=ax.transAxes, fontsize=fs*mult,
#             verticalalignment='top', bbox=props,rotation=0,color=color, rotation_mode="anchor")  
#     # plt.axhline(y=0,linestyle='-',c='k')
#     lo=0.8 ; hi=3.6
#     plt.plot([lo,hi], [lo,hi], '-',c='k')
#     plt.xlim(0.9,1.2)    

#     # plt.xlim(lo,hi)    
#     # plt.ylim(lo,hi)    
#     plt.ylabel(ytit)    
#     plt.xlabel(xtit)    
#     plt.legend()

# du_T=0

# if du_T:
#     x=df_interpol['ShortwaveRadiationDown_Cor(W/m2)']
#     y1=df_interpol[varname1[0]]-df_interpol[PROMICE_temperature]
#     y2=df_interpol[varname1[1]]-df_interpol[PROMICE_temperature]
#     fig, ax = plt.subplots(figsize=(10,10))
#     color='b'
#     plt.scatter(x,y1,marker='s',color=color,label= 'GC-Net 1 minus '+PROMICE_temperature_name)
#     y=y1
#     v=np.where((np.isfinite(x))&(np.isfinite(y)))
#     b, m = polyfit(x[v[0]], y[v[0]], 1)
#     xx=[np.min(x[v[0]]),np.max(x[v[0]])]
#     yy=[b + m * xx[0],b + m * xx[1]]
#     plt.plot(xx, yy, '--',c=color)
#     print("range ",yy[1]-yy[0])
#     print("range min ",yy[1])
#     print("range max ",yy[0])

#     k=0
#     RMSD,R,bias,N =gnl.comp_stats(df_interpol[varname1[k]],df_interpol[PROMICE_temperature])
#     # print("RMSD",varname1[k],PROMICE_temperature,RMSD)
#     # print("R",varname1[k],PROMICE_temperature,R)
#     # print("bias",varname1[k],PROMICE_temperature,bias)
#     # print("N",varname1[k],PROMICE_temperature,N)    
    
#     cc=0
#     mult=1
#     xx0=1.01 ; yy0=0.99 ; dy=-0.15
#     props = dict(boxstyle='round', facecolor='w', alpha=0.5,edgecolor='w')
#     ax.text(xx0,yy0, 'N: '+str('%.0f'%N)+'\n'
#             +'correlation: '+str('%.3f'%R)+'\n'
#             +'bias: '+str('%.2f'%bias)+'\n'
#             +'RMSD: '+str('%.2f'%RMSD)+'\n'
#             ,transform=ax.transAxes, fontsize=fs*mult,
#             verticalalignment='top', bbox=props,rotation=0,color=color, rotation_mode="anchor")  

#     color='r'    
#     plt.scatter(x,y2,marker='s',color=color,label= 'GC-Net 2 minus '+PROMICE_temperature_name)
#     y=y2
#     v=np.where((np.isfinite(x))&(np.isfinite(y)))
#     b, m = polyfit(x[v[0]], y[v[0]], 1)
#     yy=[b + m * xx[0],b + m * xx[1]]
#     plt.plot(xx, yy, '--',c=color)

#     k=1
#     RMSD,R,bias,N =gnl.comp_stats(df_interpol[varname1[k]],df_interpol[PROMICE_temperature])
#     print("RMSD",varname1[k],PROMICE_temperature,RMSD)
#     print("R",varname1[k],PROMICE_temperature,R)
#     print("bias",varname1[k],PROMICE_temperature,bias)
#     print("N",varname1[k],PROMICE_temperature,N)    
    
#     cc+=1
    
#     yy0+=dy*cc
#     ax.text(xx0,yy0, 'N: '+str('%.0f'%N)+'\n'
#             +'correlation: '+str('%.3f'%R)+'\n'
#             +'bias: '+str('%.2f'%bias)+'\n'
#             +'RMSD: '+str('%.2f'%RMSD)+'\n'
#             ,transform=ax.transAxes, fontsize=fs*mult,
#             verticalalignment='top', bbox=props,rotation=0,color=color, rotation_mode="anchor")  
#     plt.axhline(y=0,linestyle='-',c='k')
#     plt.ylabel('air temperature difference, °C')    
#     plt.xlabel('shortwave downward, W m$^{-2}$')    
#     plt.legend()


# #%%
# wo_fig=0
# mult=0.7
# txt2='PROMICE' ; figure_name=station#+'_temp'
# fig, ax = plt.subplots(4,1,figsize=(10,12))
# # fig.subplots_adjust(hspace=0.01)
# # fig.suptitle(tit)

# i=0
# j = 0
# # ax[0].plot(df_interpol.index, df_interpol[varname1[i]],
# #               'b',label= 'GC-Net')
# # ax[0].plot(df_interpol.index, df_interpol[varname2[i]],
# #               'r',label= 'PROMICE')
# # ax[0].title('EGP')
# cc=0#---------------------------------------------------------------- T
# ax[0].set_title(tit)#+' '+PROMICE_temperature_name)
# ax[cc].plot(df_interpol.index, df_interpol[varname1[0]],
#             '.--',color='b',label= 'GC-Net 1')
# # ax[cc].plot(df_interpol.index, df_interpol[varname1[1]],
# #             '.--',color='r',label= 'GC-Net 2')
# ax[cc].plot(df_interpol.index, df_interpol[PROMICE_temperature],
#             '.--',color='k',label= 'PROMICE PT100')
# # ax[cc].plot(df_interpol.index, df_interpol['AirTemperatureHygroClip(C)'],
# #             '.--',color='grey',label= 'PROMICE Hygroclip')
# ax[cc].get_xaxis().set_visible(False)
# ax[cc].legend(prop={'size': fs*mult})
# ax[cc].set_ylabel('air temperature, °C')

# cc+=1#---------------------------------------------------------------- T diff
# ax[cc].plot(df_interpol.index, df_interpol[varname1[0]]-df_interpol[PROMICE_temperature],
#             '.--',color='b',label= 'GCN 1 - PROMICE PT100')
# RMSD,R,bias,N =gnl.comp_stats(df_interpol[varname1[0]],df_interpol[PROMICE_temperature])
# print("RMSD",varname1[0],PROMICE_temperature,RMSD)
# print("R",varname1[0],PROMICE_temperature,R)
# print("bias",varname1[0],PROMICE_temperature,bias)
# print("N",varname1[0],PROMICE_temperature,N)

# # ax[cc].plot(df_interpol.index, df_interpol[varname1[1]]-df_interpol[PROMICE_temperature],
# #             '.--',color='r',label= 'GCN 2 - PROMICE PT100')
# # ax[cc].plot(df_interpol.index, df_interpol['AirTemperatureHygroClip(C)']-df_interpol[PROMICE_temperature],
# #             '.--',color='k',label= 'PROMICE Hygroclip - PROMICE PT100')
# # ax[cc].plot(df_interpol.index, df_interpol['AirTemperatureHygroClip(C)']-df_interpol[PROMICE_temperature],
#             # '.--',color='r',label= 'PROMICE Hygroclip - PROMICE PT100')
# ax[cc].get_xaxis().set_visible(False)
# ax[cc].axhline(y=0,linestyle='--',c='k')
# ax[cc].legend(prop={'size': fs*mult})
# ax[cc].set_ylabel('air temperature\ndifference, °C')

# cc+=1 #---------------------------------------------------------------- radiation
# color='darkorange'
# ax[cc].plot(df_interpol.index, df_interpol['ShortwaveRadiationDown_Cor(W/m2)'],
#             '.--',color=color,label= 'SWD')
# ax[cc].set_ylabel('SWD, W m$^{-2}$', color=color)
# ax[cc].get_xaxis().set_visible(False)
# ax[cc].tick_params(axis='y', labelcolor=color)
# ax[cc].legend(loc=2,prop={'size': fs*mult})
# ax[cc] = ax[cc].twinx()  # instantiate a second axes that shares the same x-axis    
# color = 'g'
# ax[cc].set_ylabel('sin', color=color)  # we already handled the x-label with ax1
# # ax[cc].plot(df_interpol.index, df_interpol['LongwaveRadiationDown(W/m2)'],
# #             '.--',color=color,label= 'LWD')
# ax[cc].plot(df_interpol.index, (df_interpol['LongwaveRadiationDown(W/m2)']-df_interpol['LongwaveRadiationUp(W/m2)']),
#             '.--',color=color,label= 'LW net')
# ax[cc].tick_params(axis='y', labelcolor=color)
# ax[cc].set_ylabel('infrared irradiance, W m$^{-2}$')
# ax[cc].legend(loc=1,prop={'size': fs*mult})


# ax[cc].get_xaxis().set_visible(False)
# cc+=1 #---------------------------------------------------------------- wind

# ax[cc].plot(df_interpol.index, df_interpol['VW1'],'grey',label='GCN 1')
# ax[cc].plot(df_interpol.index, df_interpol['VW2'],'k',label='GCN 2')
# ax[cc].plot(df_interpol.index, df_interpol['WindSpeed(m/s)'],'r',label='PROMICE')
# ax[cc].set_ylabel('wind speed, m s$^{-1}$')
# # ax[cc].legend()
# ax[cc].legend(loc=1,prop={'size': fs*mult})

# # ax[cc].set_xticks(rotation=45)

# plt.setp( ax[cc].xaxis.get_majorticklabels(), rotation=45, ha="right", rotation_mode="anchor") 


# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.subplots_adjust(hspace=0)

# plt.show()
 
# figpath='./Figs/'
# if wo_fig:fig.savefig(figpath+fignam+'png',bbox_inches='tight', dpi=200)
