# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 22:47:09 2024

@author: yang8460
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd
import KGDA_util as util
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['figure.dpi'] = 300
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable

def visualizeCountyLevel(dataDic,FIPSlist,crop='corn',
                  crs = "EPSG:2163",term='',vmin=None, vmax=None,descrip='',col=4,stats=None):
    # load data
    state_shp = gpd.read_file('I:/shp/shp_US_states_continental.shp')
    state_shp = state_shp.to_crs(crs)
    county_shp = gpd.read_file('I:/shp/shp_US_Counties_TIGER_continental_clipped.shp')
    county_shp['PARA'] = np.nan
    county_shp = county_shp.to_crs(crs)

    # replace the county-level data   
    for i in FIPSlist:
        county_shp.loc[county_shp['GEOID']==str(i),'PARA'] = dataDic[i]
        
    
    ax = plt.subplot(2,col,index)   
    # compass
    ax.text(0.055, 0.76, s='N', fontsize=18,transform=ax.transAxes)
    ax.arrow(0.08, 0.95, 0, 0.01, length_includes_head=True,
              head_width=0.05, head_length=0.12, overhang=.1, facecolor='k',transform=ax.transAxes)

    # note
    ax.text(0.065, 0.05, s=descrip, fontsize=18,transform=ax.transAxes)
    if stats is not None:
        ax.text(0.58, 0.91, s=r'$R^2$= %s'%str(stats[0])[:5], fontsize=12,transform=ax.transAxes,
                backgroundcolor = 'w')
        ax.text(0.58, 0.82, s=r'$RMSE$= %s'%str(stats[1])[:5], fontsize=12,transform=ax.transAxes,
                backgroundcolor = 'w')
        ax.text(0.58, 0.73, s=r'$Bias$= %s'%str(stats[2])[:5], fontsize=12,transform=ax.transAxes,
                backgroundcolor = 'w')
        
    if crs == "EPSG:3395":
        plt.xlim([-1.1*1e7,-0.9*1e7])
        plt.ylim([4.1*1e6,5.8*1e6])
    elif crs == "EPSG:2163":
        plt.xlim([-0.2*1e6, 1.7*1e6])
        plt.ylim([-1.1*1e6,0.5*1e6])
    # plot para distribution
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1) # the cax is used to minimize the white gap 
    county_shp.plot(ax=ax,column='PARA',legend=True, cax=cax,vmin=vmin, vmax=vmax, cmap='RdYlGn')#cmap = 'coolwarm')
    plt.title(term)
        
    # plot background  
    state_shp.plot(ax=ax,facecolor="none", edgecolor="k",linewidth=0.5)
    # ax.set_axis_off()
    
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

    return fig

def stat(x,y):
    x,y = util.removeNaN(x,y) 
    R2 = np.corrcoef(x, y)[0, 1] ** 2
    RMSE = (np.sum((y - x) ** 2) / len(y)) ** 0.5
    Bias = np.mean(y) - np.mean(x)
    return R2, RMSE, Bias
 
if __name__ == '__main__':
    root = r'F:\countyYieldResult\EnsembleSigma\county_yield_parallel_epoch30-batch256_maskc3c4_obsInterval8_countyMerge_intevalAcc_caseobs_GPP_globalf-230115-114432'
    carbonflux = util.load_object('%s/carbonCycle.pkl'%root)
    fig = plt.figure(figsize=(19,7))
    index = 1
    siteList = util.cornBeltCountyDA()
   
    # NASS yield
    NASS_Path = 'F:/MidWest_counties/Yield'
    NASS_yield = util.yieldValidation(NASS_Path)
                
    col=4
    datadic_all = {}
    for crop in ['corn','soy']:
        datadic_all[crop] = {}
        for descrip,case in zip(['GPP','Yield','Reco','NBP'],[0,2,1,3]):
            stats=None
            if case == 0: # GPP
                if crop == 'corn':
                    GPP_stat = pd.read_csv(r'F:\MidWest_counties\GPP\GPP_SumGS_corn.csv')
                else:
                    GPP_stat = pd.read_csv(r'F:\MidWest_counties\GPP\GPP_SumGS_soybean.csv')
                
                dataDic = {}
                GPP_mean = []
                for s in siteList:
                    target = np.stack([t[:365] if t is not None else [np.nan]*365 for t in carbonflux[crop][case][s]])[3:,:]
                    target_mean = np.nanmean(target,axis=0)
                    target_seanson_sum = np.sum(target_mean[151:243])  # Jun.1- Sep.1)
                
                    dataDic[s] = target_seanson_sum
                    GPP_mean.append(target_seanson_sum)
                GPP_SLOPE_mean = np.nanmean(GPP_stat[siteList].values[3:,:], axis=0)
                stats  = stat(x=GPP_SLOPE_mean ,y=GPP_mean)
            if case == 2: # yield
            
                if crop == 'corn':
                    # prediction
                    yield_data = pd.read_csv('%s/yield_corn.csv'%root)
                    # nass yield
                    df_yield_nass = NASS_yield.yield_NASS_corn
                    coef = NASS_yield.coef_C2BUacre(0)
                else:
                    # prediction
                    yield_data = pd.read_csv('%s/yield_soybean.csv'%root)
                    # nass yield
                    df_yield_nass = NASS_yield.yield_NASS_soybean
                    coef = NASS_yield.coef_C2BUacre(1)
                
                yearRange = [2000,2020]
                yearRange_t = [t for t in range(yearRange[0],yearRange[1]+1)]
                loc = [i for i,t in enumerate(df_yield_nass.Year.tolist()) if t in yearRange_t]
                df_yield_nass_ = df_yield_nass.iloc[loc]
                if crop == 'soy':
                    yield_df, _ = util.individualTrend(df_yield_nass_,root,crop='soybean',mode='intevalAcc',interval=3,
                                                                     yearRange_t=yearRange_t,coef=coef)
                else:
                    yield_df, _ = util.individualTrend(df_yield_nass_,root,crop=crop,mode='intevalAcc',interval=3,
                                                                     yearRange_t=yearRange_t,coef=coef)
                yield_mean = np.nanmean(yield_df[siteList].values[3:,:]/coef, axis=0)
                yield_nass_mean = np.nanmean(df_yield_nass[siteList].values[3:,:]/coef, axis=0)
                stats  = stat(x=yield_nass_mean,y=yield_mean)
                dataDic = {}
                for s,t in zip(siteList,yield_mean):
                    dataDic[s] = t
                    
            elif case == 3:
                dataDic = {}
                for s,i in  datadic_all[crop]['GPP'].items():
                    dataDic[s] = i-datadic_all[crop]['Reco'][s] - datadic_all[crop]['Yield'][s]
            else:        
                dataDic = {}
                for s in siteList:
                    target = np.stack([t[:365] if t is not None else [np.nan]*365 for t in carbonflux[crop][case][s]])[3:,:]
                    target_mean = np.nanmean(target,axis=0)
                    target_seanson_sum = np.sum(target_mean[151:243])  # Jun.1- Sep.1)
                
                    dataDic[s] = target_seanson_sum
                            
            datadic_all[crop][descrip] =  dataDic
            if crop == 'corn':
                fig_site_post2 = visualizeCountyLevel(dataDic,FIPSlist=siteList, crop='Corn',
                                          crs = "EPSG:2163", term='%s $(gC/m^2)$'%descrip,
                                          descrip=descrip,col=col,stats=stats)
            else:
                fig_site_post2 = visualizeCountyLevel(dataDic,FIPSlist=siteList, crop='Soybean',
                                          crs = "EPSG:2163", term='%s $(gC/m^2)$'%descrip,
                                          descrip=descrip,col=col,stats=stats)
            index+=1  
    
    fig.text(0.1, 0.73, 'Corn', va='center', rotation='vertical',fontsize=20)
    fig.text(0.1, 0.32, 'Soybean', va='center', rotation='vertical',fontsize=20)