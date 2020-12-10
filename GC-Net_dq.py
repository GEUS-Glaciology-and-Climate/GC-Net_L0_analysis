# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit
from glob import glob
import datetime

base_dir = '/Users/jason/0_dat/GCNET_L0/_all_c/'
files = sorted(glob(base_dir+'*'))

n_stations=len(files)

fn='/Users/jason/Dropbox/temp/GCNet/ancillary/GCN info ca.2000.csv'
fn='/Users/jason/Dropbox/AWS/GCNET/ancillary/StationDescription ca.2000 GCN.csv'
info_all = pd.read_csv(fn)
info_all.columns

fn='/Users/jason/Dropbox/AWS/GCNET/ancillary/varnames_all.txt'
info=pd.read_csv(fn, header=None, delim_whitespace=True)
info.columns=['id','varnam']
print(info.columns)
print(info.varnam)

varsx=info.varnam[3:47]
varsx=info.varnam[3:29]
n_vars=len(varsx)

#%% Relative humidity tools
def RH_water2ice(RH, T):
    # switch ONLY SUBFREEZING timesteps to with-regards-to-ice

    Lv = 2.5001e6  # H2O Vaporization Latent Heat (J/kg)
    Ls = 2.8337e6  # H2O Sublimation Latent Heat (J/kg)
    Rv = 461.5     # H2O Vapor Gaz constant (J/kg/K)
    ind = T < 0
    TCoeff = 1/273.15 - 1/(T+273.15)
    Es_Water = 6.112*np.exp(Lv/Rv*TCoeff)
    Es_Ice = 6.112*np.exp(Ls/Rv*TCoeff)
    RH_out = RH.copy()
    RH_out[ind] = RH[ind] * Es_Water[ind]/Es_Ice[ind] 
    return RH_out

def RH_ice2water(RH, T):
    # switch ALL timestep to with-regards-to-water
    RH = np.array(RH)
    Lv = 2.5001e6  # H2O Vaporization Latent Heat (J/kg)
    Ls = 2.8337e6  # H2O Sublimation Latent Heat (J/kg)
    Rv = 461.5     # H2O Vapor Gaz constant (J/kg/K)
    ind = T < 0
    TCoeff = 1/273.15 - 1/(T+273.15)
    Es_Water = 6.112*np.exp(Lv/Rv*TCoeff)
    Es_Ice = 6.112*np.exp(Ls/Rv*TCoeff)
    RH_out = RH.copy()
    
    # T_100 = 373.15
    # T_0 = 273.15
    # T = T +T_0
    # # GOFF-GRATCH 1945 equation
    #    # saturation vapour pressure above 0 C (hPa)
    # Es_Water = 10**(  -7.90298*(T_100/T - 1) + 5.02808 * np.log(T_100/T) 
    #     - 1.3816E-7 * (10**(11.344*(1-T/T_100))-1) 
    #     + 8.1328E-3*(10**(-3.49149*(T_100/T-1)) -1.) + np.log(1013.246) )
    # # saturation vapour pressure below 0 C (hPa)
    # Es_Ice = 10**(  -9.09718 * (T_0 / T - 1.) - 3.56654 * np.log(T_0 / T) + 
    #              0.876793 * (1. - T / T_0) + np.log(6.1071)  )   
    
    RH_out[ind] = RH[ind] / Es_Water[ind]*Es_Ice[ind] 

    return RH_out

def RH_ice2water2(RH, T):
    # switch ALL timestep to with-regards-to-water
    RH = np.array(RH)
    # Lv = 2.5001e6  # H2O Vaporization Latent Heat (J/kg)
    # Ls = 2.8337e6  # H2O Sublimation Latent Heat (J/kg)
    # Rv = 461.5     # H2O Vapor Gaz constant (J/kg/K)
    ind = T < 0
    # TCoeff = 1/273.15 - 1/(T+273.15)
    # Es_Water = 6.112*np.exp(Lv/Rv*TCoeff)
    # Es_Ice = 6.112*np.exp(Ls/Rv*TCoeff)
    RH_out = RH.copy()
    
    T_100 = 373.15
    T_0 = 273.15
    T = T +T_0
    # GOFF-GRATCH 1945 equation
        # saturation vapour pressure above 0 C (hPa)
    Es_Water = 10**(  -7.90298*(T_100/T - 1) + 5.02808 * np.log10(T_100/T) 
        - 1.3816E-7 * (10**(11.344*(1-T/T_100))-1) 
        + 8.1328E-3*(10**(-3.49149*(T_100/T-1)) -1.) + np.log10(1013.246) )
    # saturation vapour pressure below 0 C (hPa)
    Es_Ice = 10**(  -9.09718 * (T_0 / T - 1.) - 3.56654 * np.log10(T_0 / T) + 
                  0.876793 * (1. - T / T_0) + np.log10(6.1071)  )   
    
    RH_out[ind] = RH[ind] / Es_Water[ind]*Es_Ice[ind] 

    return RH_out

# def RH_ice2water3(RH, T):
#     # switch ALL timestep to with-regards-to-water
#     RH = np.array(RH)
#     # Lv = 2.5001e6  # H2O Vaporization Latent Heat (J/kg)
#     # Ls = 2.8337e6  # H2O Sublimation Latent Heat (J/kg)
#     # Rv = 461.5     # H2O Vapor Gaz constant (J/kg/K)
#     ind = T < 0
#     # TCoeff = 1/273.15 - 1/(T+273.15)
#     # Es_Water = 6.112*np.exp(Lv/Rv*TCoeff)
#     # Es_Ice = 6.112*np.exp(Ls/Rv*TCoeff)
#     RH_out = RH.copy()
    
#     T_100 = 373.15
#     T_0 = 273.15
#     T = T +T_0
#    # saturation vapour pressure above 0 C (hPa)
#     Es_Water = 10**(  10.79574*(1 - T_100/T) + 5.028 * np.log10(T / T_100)
#                     + 1.50475E-4 * (1 - 10**(-8.2969 * (T/T_100 - 1)))
#                     + 0.42873E-3*(10**(4.76955*(1 - T_100/T)) -1.) +  0.78614 + 2.0 )

#     Es_Ice = 10**( -9.09685 * (T_0 / T - 1.) - 3.56654 * np.log10(T_0 / T) +
#                   0.87682 * (1. - T / T_0) + 0.78614   )
#     RH_out[ind] = RH[ind] / Es_Water[ind]*Es_Ice[ind] 

#     return RH_out

def RH2SpecHum(RH, T, pres):
    # Note: RH[T<0] needs to be with regards to ice
    
    Lv = 2.5001e6  # H2O Vaporization Latent Heat (J/kg)
    Ls = 2.8337e6  # H2O Sublimation Latent Heat (J/kg)
    Rv = 461.5     # H2O Vapor Gaz constant (J/kg/K)
    es = 0.622
    
    TCoeff = 1/273.15 - 1/(T+273.15)
    Es_Water = 6.112*np.exp(Lv/Rv*TCoeff)
    Es_Ice = 6.112*np.exp(Ls/Rv*TCoeff)
    
    es_all = Es_Water.copy()
    es_all[T < 0] = Es_Ice[T < 0] 
    
    # specific humidity at saturation
    q_sat = es * es_all/(pres-(1-es)*es_all)

    # specific humidity
    q = RH * q_sat /100
    return q

def SpecHum2RH(q, T, pres):
    # Note: RH[T<0] will be with regards to ice
    
    Lv = 2.5001e6  # H2O Vaporization Latent Heat (J/kg)
    Ls = 2.8337e6  # H2O Sublimation Latent Heat (J/kg)
    Rv = 461.5     # H2O Vapor Gaz constant (J/kg/K)
    es = 0.622
    
    TCoeff = 1/273.15 - 1/(T+273.15)
    Es_Water = 6.112*np.exp(Lv/Rv*TCoeff)
    Es_Ice = 6.112*np.exp(Ls/Rv*TCoeff)
    
    es_all = Es_Water
    es_all[T < 0] = Es_Ice
    
    # specific humidity at saturation
    q_sat = es * es_all/(pres-(1-es)*es_all)

    # relative humidity
    RH = q / q_sat *100
    return RH


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
plt.rcParams['figure.figsize'] = 12, 12.3

path0='/Users/jason/Dropbox/temp/' # set data path for your system

for i,fn in enumerate(files[0:1]): # just the first station
# for i,fn in enumerate(files): # all sations
    stnum=fn.split('/')[-1][0:2]
    df_gc=pd.read_csv(path0+'GCNet/hourly_data/'+stnum+'.csv')
    print(df_gc.columns)
    df_gc['time'] = pd.to_datetime(df_gc.date)
    # select only year 
    year_choice='2010'
    df_gc = df_gc.loc[df_gc['time']<year_choice+'-12-31',:] 
    df_gc = df_gc.loc[df_gc['time']>year_choice+'-01-01',:]
    
    #compute humidity variables
    df_gc['RH1_w'] = RH_ice2water(df_gc['RH1'] ,df_gc['AirTemp1(CS500)'])
    df_gc['RH2_w'] = RH_ice2water(df_gc['RH2'] ,df_gc['AirTemp2CS500)'])
    df_gc['SpecHum1'] = RH2SpecHum(df_gc['RH1'], df_gc['AirTemp1(CS500)'], df_gc['AtmosPressure'] )*1000
    df_gc['SpecHum2'] = RH2SpecHum(df_gc['RH2'], df_gc['AirTemp2CS500)'], df_gc['AtmosPressure'] )*1000
    
    # exclude impossible values
    df_gc['SpecHum2'][df_gc['SpecHum2']<0]=np.nan
    df_gc['SpecHum1'][df_gc['SpecHum1']<0]=np.nan

    fig, ax = plt.subplots(2,1)

    cc=0
    ax[cc].set_title('GC-Net specific humidity')
    ax[cc].plot(df_gc['time'],df_gc['SpecHum1'])
    ax[cc].plot(df_gc['time'],df_gc['SpecHum2'])

    cc+=1

    ax[cc].set_title('GC-Net specific humidity vertical gradient')
    ax[cc].plot(df_gc['time'],df_gc['SpecHum1']-df_gc['SpecHum2'])
    # ax[cc].plot(df_gc['time'],)


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
