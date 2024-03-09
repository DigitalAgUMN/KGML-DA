# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 14:57:32 2023

@author: yang8460
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 22:07:51 2022

@author: yang8460

https://towardsdatascience.com/bayesian-hierarchical-modeling-in-pymc3-d113c97f5149

problem: 
    - there are 20 (year) groups or 600+ (site) groups
    - want to estimate the posterior dist for each group
    
note:
    - MCMC is trying to handle situation when the prior or likelihood follow a arbitrary dist,
    so the posterior will be a ugly dist and MCMC can do sampling from a ugly dist and the giving
    the hist of the posterioir dist
    
    - hierarchical bayesain modelling, also called partially pooled, trying to
    borrow knowledge from other groups/samples to estimated the dist for a baised 
    (with limited samples, or have outlier) samples/groups
    
    v2: for different cases, discard unpooled model
    v3: mu_g follow a normal dist
"""

import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import arviz as az
import matplotlib
from pdf2image import convert_from_path
import pandas as pd
import KGDA_util as util

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['figure.dpi'] = 300
poppler_path = 'F:/poppler-22.12.0/Library/bin'


import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable

def visualizeCountyLevel(dataDic,FIPSlist,crop='corn',
                  crs = "EPSG:2163",term='',vmin=None, vmax=None):
    # load data
    state_shp = gpd.read_file('E:/shp/shp_US_states_continental.shp')
    state_shp = state_shp.to_crs(crs)
    county_shp = gpd.read_file('E:/shp/shp_US_Counties_TIGER_continental_clipped.shp')
    county_shp['PARA'] = np.nan
    county_shp = county_shp.to_crs(crs)

    # replace the county-level data   
    for i in FIPSlist:
        county_shp.loc[county_shp['GEOID']==i,'PARA'] = dataDic[i]
        
    # plot background   
    fig,ax = plt.subplots(figsize=(10,7))   
    state_shp.plot(ax=ax,facecolor="none", edgecolor="grey")
    # ax.set_axis_off()
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    
    if crs == "EPSG:3395":
        plt.xlim([-1.1*1e7,-0.9*1e7])
        plt.ylim([4.1*1e6,5.8*1e6])
    elif crs == "EPSG:2163":
        plt.xlim([-0.2*1e6, 1.7*1e6])
        plt.ylim([-1.1*1e6,0.5*1e6])
    
    # compass
    ax.text(0.065, 0.79, s='N', fontsize=20,transform=ax.transAxes)
    ax.arrow(0.08, 0.95, 0, 0.01, length_includes_head=True,
              head_width=0.05, head_length=0.12, overhang=.1, facecolor='k',transform=ax.transAxes)

    # note
    ax.text(0.065, 0.05, s='%s of %s'%(term,crop), fontsize=20,transform=ax.transAxes)
    
    # plot para distribution
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    county_shp.plot(ax=ax,column='PARA',legend=True, cax=cax,vmin=vmin, vmax=vmax)
    plt.title(term)
    return fig

def removeNaN(obs,pre,coef=1):
    x = np.array(obs)
    y = np.array(pre)*coef
    
    Loc = (1 - (np.isnan(x) | np.isnan(y)))
    x_ = x[Loc==1]
    y_ = y[Loc==1]
    return x_,y_

def showStructure(model):
    graphviz = pm.model_to_graphviz(model)
    graphviz.render('tmp')  
    img = convert_from_path('tmp.pdf',poppler_path=poppler_path,dpi=300)
    fig = plt.figure()
    plt.imshow(img[0])
    plt.axis("off")
    return fig

def getIndexMat(data):
    rowIndex = np.tile(np.array([t for t in range(data.shape[0])]),(data.shape[1],1)).T
    colIndex = np.tile(np.array([t for t in range(data.shape[1])]),(data.shape[0],1))
    
    data_flat = np.reshape(data,-1)
    rowIndex_flat = np.reshape(rowIndex,-1)
    colIndex_flat = np.reshape(colIndex,-1)
    
    return data_flat,rowIndex_flat,colIndex_flat

def postStatic(post):   
    # post_mu = np.squeeze(np.mean(post,axis=1))
    # post_sig = np.squeeze(np.std(post,axis=1))  
    # percent_97 = np.squeeze(np.percentile(post, 97, axis=1))
    # percent_3 = np.squeeze(np.percentile(post, 3, axis=1))
    post_mu = np.mean(np.mean(post,axis=1),axis=0)
    post_sig = np.mean(np.std(post,axis=1),axis=0)
    percent_97 = np.mean(np.percentile(post, 97, axis=1),axis=0)
    percent_3 = np.mean(np.percentile(post, 3, axis=1),axis=0)
    
    return post_mu,post_sig,percent_97,percent_3

def nassYield(crop):
    # obs NASS yield
    NASS_Path = 'F:/MidWest_counties/Yield'
    NASS_yield = util.yieldValidation(NASS_Path)
    if crop == 'corn':
        df_yield_nass = NASS_yield.yield_NASS_corn
        coef = NASS_yield.coef_C2BUacre(0)
    else:
        df_yield_nass = NASS_yield.yield_NASS_soybean
        coef = NASS_yield.coef_C2BUacre(1)
    df_yield_nass.drop(['Unnamed: 0'],inplace=True,axis=1)
    return df_yield_nass, coef
def individualTrend(df_yield_nass,yield_df,mode,interval,yearRange_t,coef=1):
    # calculate the trend of NASS yield
    yield_df.drop(['Unnamed: 0'],inplace=True,axis=1) 
    paraDic = {}
    validFIPS = list(set((df_yield_nass.columns[1:]).intersection(set(yield_df.columns[1:]))))
    validFIPS.sort()
    for t in validFIPS:        
        trend = df_yield_nass[t][df_yield_nass[t]>0].values
        trendYear = df_yield_nass['Year'][df_yield_nass[t]>0].values 
        para = np.polyfit(trendYear, trend, 1)
        paraDic[t] = para
        # y_fit = np.polyval(para, trendYear)  #
        # plt.figure(figsize = (12,5))
        # plt.plot(trendYear, y_fit, 'r')
        # plt.plot(trendYear, trend, 'b.')        

    delta_y = np.array([paraDic[t][0] for t in validFIPS])
    # correct predicted yield trend
    nodeList = [t for t in range(2000,2019,interval)]   
    recorrected = []
    for year in yearRange_t:        
        # find the end of calibraion set
        if year< nodeList[1]:
             tmp = np.array(yield_df[yield_df['Year']==year])[:,1:]*coef
             recorrected.append(tmp)       
        else:
            if year >= 2018:
                end=2018
            else:
                for i in range(len(nodeList)-1):
                    if (nodeList[i]<=year)&(nodeList[i+1]>year):
                        end = nodeList[i]
                        break
            # the midyear of calibration set           
            if mode=='previousYear':
                tmp = np.array(yield_df[yield_df['Year']==year])[:,1:]*coef+delta_y
            else:
                if mode=='intevalAcc':
                    midYear = 0.5*(2000+end)
                elif mode=='inteval':
                    midYear = 0.5*(end-interval+end)
                elif mode=='default':
                    midYear = 2010
                    
                tmp = np.array(yield_df[yield_df['Year']==year])[:,1:]*coef+(year-midYear)*delta_y
            recorrected.append(tmp)
    keys = yield_df.columns.tolist()[1:]
    df_new = pd.DataFrame(np.concatenate(recorrected,axis=0),columns=keys)
    df_new.insert(0,'Year',yearRange_t)        
                    
    return df_new, yield_df

def casePathes():
    modeDic = {}
    outFolderDic = {}
    # openloop
    mode = 'default'
    name = '%s_op'%mode
    outFolder = r'F:\countyYieldResult\test\county_yield_parallel_epoch30-batch256'+\
        '_openLoop_obsInterval8_countyMerge_%s-221113-171739'%mode
    modeDic[name] = mode
    outFolderDic[name] = outFolder
    
    # assililating current GPP
    mode = 'default'
    name = '%s_GPP'%mode
    outFolder = r'F:\countyYieldResult\test\county_yield_parallel_epoch30-batch256'+\
        '_maskc3c4_obsInterval8_countyMerge_%s_caseobs_GPP-221115-114012'%mode
    modeDic[name] = mode
    outFolderDic[name] = outFolder
    
    # assililating current ET
    mode = 'default'
    name = '%s_ET'%mode
    outFolder = r'F:\countyYieldResult\test\county_yield_parallel_epoch30-batch256'+\
        '_maskc3c4_obsInterval8_countyMerge_%s_caseobs_ET_adaptionMeanGS_globalf_R0.16-221115-005442'%mode
    modeDic[name] = mode
    outFolderDic[name] = outFolder
    
    # assililating current LAI
    mode = 'default'
    name = '%s_LAI'%mode
    outFolder = r'F:\countyYieldResult\test\county_yield_parallel_epoch30-batch256'+\
        '_maskc2c3_obsInterval8_countyMerge_%s_caseobs_LAI_adaption_globalf_R0.02-221115-101216'%mode
    modeDic[name] = mode
    outFolderDic[name] = outFolder
    
    # assililating current GPP & ET
    mode = 'default'
    name = '%s_GPP_ET'%mode
    outFolder = r'F:\countyYieldResult\test\county_yield_parallel_epoch30-batch256'+\
        '_maskc3c4_obsInterval8_countyMerge_%s_caseobs_GPP_ET_adaptionMeanGS_globalf_R0.04-221115-120447'%mode
    modeDic[name] = mode
    outFolderDic[name] = outFolder
    
    # assililating current GPP & ET & LAI
    mode = 'default'
    name = '%s_GPP_ET_LAI'%mode
    outFolder = r'F:\countyYieldResult\test\county_yield_parallel_epoch30-batch256'+\
        '_maskc3_obsInterval8_countyMerge_%s_caseobs_GPP_LAI_ET_R0.02_0.04-221115-124515'%mode
    modeDic[name] = mode
    outFolderDic[name] = outFolder
       
    # calibration_acc - openloop
    mode = 'intevalAcc'
    name = '%s_op'%mode
    outFolder = r'F:\countyYieldResult\test\county_yield_parallel_epoch30-batch256'+\
        '_openLoop_obsInterval8_countyMerge_intevalAcc_caseobs_-221115-153202'
    modeDic[name] = mode
    outFolderDic[name] = outFolder
    
    # calibration_acc - GPP
    mode = 'intevalAcc'
    name = '%s_GPP'%mode
    outFolder = r'F:\countyYieldResult\test\county_yield_parallel_epoch30-batch256'+\
        '_maskc3c4_obsInterval8_countyMerge_intevalAcc_caseobs_GPP-221115-154552'
    modeDic[name] = mode
    outFolderDic[name] = outFolder
    
    # calibration_acc - LAI
    mode = 'intevalAcc'
    name = '%s_LAI'%mode
    outFolder = r'F:\countyYieldResult\test\county_yield_parallel_epoch30-batch256'+\
        '_maskc2c3_obsInterval8_countyMerge_intevalAcc_caseobs_LAI_globalf_R0.02-221115-160455'
    modeDic[name] = mode
    outFolderDic[name] = outFolder
    
    # calibration_acc - ET
    mode = 'intevalAcc'
    name = '%s_ET'%mode
    outFolder = r'F:\countyYieldResult\test\county_yield_parallel_epoch30-batch256'+\
        '_maskc3c4_obsInterval8_countyMerge_intevalAcc_caseobs_ET_adaptionMeanGS_globalf_R0.08-221115-163249'
    modeDic[name] = mode
    outFolderDic[name] = outFolder
    
    # calibration_acc - GPP&ET
    mode = 'intevalAcc'
    name = '%s_GPP_ET'%mode
    outFolder = r'F:\countyYieldResult\test\county_yield_parallel_epoch30-batch256'+\
        '_maskc3c4_obsInterval8_countyMerge_intevalAcc_caseobs_GPP_ET_adaptionMeanGS_globalf_R0.08-221115-165529'
    modeDic[name] = mode
    outFolderDic[name] = outFolder
    
    # calibration_acc - GPP&ET&LAI
    mode = 'intevalAcc'
    name = '%s_GPP_ET_LAI'%mode
    outFolder = r'F:\countyYieldResult\test\county_yield_parallel_epoch30-batch256'+\
        '_maskc3_obsInterval8_countyMerge_intevalAcc_caseobs_GPP_LAI_ET_globalf_R0.02_0.08-221115-172805'
    modeDic[name] = mode
    outFolderDic[name] = outFolder
    
    # calibration_interval - 1 - op
    mode = 'inteval'
    name = '%s_1_op'%mode
    outFolder = r'F:\countyYieldResult\test\county_yield_parallel_epoch30-batch256'+\
        '_openLoop_obsInterval8_countyMerge_previousYear_caseobs__globalf-230110-170220'
    modeDic[name] = mode
    outFolderDic[name] = outFolder
    
    # calibration_interval - 3 - op
    mode = 'inteval'
    name = '%s_3_op'%mode
    outFolder = r'F:\countyYieldResult\test\county_yield_parallel_epoch30-batch256'+\
        '_openLoop_obsInterval8_countyMerge_inteval_caseobs__globalf-230110-170947'
    modeDic[name] = mode
    outFolderDic[name] = outFolder
    
    # calibration_interval - 1 - GPP
    mode = 'inteval'
    name = '%s_1_GPP'%mode
    outFolder = r'F:\countyYieldResult\test\county_yield_parallel_epoch30-batch256'+\
        '_maskc3c4_obsInterval8_countyMerge_previousYear_caseobs_GPP_globalf-230110-162001'
    modeDic[name] = mode
    outFolderDic[name] = outFolder
    
    # calibration_interval - 3 - GPP
    mode = 'inteval'
    name = '%s_3_GPP'%mode
    outFolder = r'F:\countyYieldResult\test\county_yield_parallel_epoch30-batch256'+\
        '_maskc3c4_obsInterval8_countyMerge_inteval_caseobs_GPP_globalf-230110-164055'
    modeDic[name] = mode
    outFolderDic[name] = outFolder
    return modeDic, outFolderDic

if __name__ == '__main__':
    modeDic, outFolderDic = casePathes()
    caseList = ['default_op','default_GPP','default_ET','default_GPP_ET','default_GPP_ET_LAI',
                'intevalAcc_op','intevalAcc_GPP','intevalAcc_ET','intevalAcc_GPP_ET',
                'intevalAcc_GPP_ET_LAI'] #'intevalAcc_GPP_ET'
    caseList = ['intevalAcc_GPP_ET', 'intevalAcc_GPP_ET_LAI']
    # re-run on 2023/1/12, revised the 97% error
    trendCorrect = True
    discard2012 = False
    saveRes = False
    root = 'H:/My Drive/ECONET_results/bayesianNet/v3'
    for case in caseList:
        if trendCorrect:
            if discard2012:
                outPath = '%s/trendCorrected_discard2012/%s'%(root,case)
            else:
                outPath = '%s/trendCorrected/%s'%(root,case)
        else:
            outPath = '%s/%s'%(root,case)
        util.mkdir(outPath)       
        
        for crop in ['corn','soybean']:
        # for crop in ['soybean']:    
            print('processing %s, %s'%(case, crop))
            ## prepare the data
            data_df = pd.read_csv('%s/yield_%s.csv'%(outFolderDic[case],crop))
            yearSpan = data_df['Year'].values
            siteList = [t for t in data_df.columns if len(t)==5]
            df_yield_nass, coef = nassYield(crop)
            nass_yield = df_yield_nass[siteList].values / coef
            
            # add trend
            if trendCorrect:
                yield_df, yield_df_origin = individualTrend(df_yield_nass,data_df,mode=modeDic[case]
                                                            ,interval=3,yearRange_t=[t for t in range(2000,2020+1)],coef=coef)
                pre_yield = yield_df[siteList].values / coef
                error = np.abs(nass_yield - pre_yield)[3:,:] # discard first three years
                
                if discard2012:
                    error = np.delete(error,9,axis=0)
            else:
                pre_yield = data_df[siteList].values
                error = np.abs(nass_yield - pre_yield)
                
            # index matrix                       
            yList,groupList_year,groupList_site = getIndexMat(error)
            Loc = (1 - np.isnan(yList))
            yList_ = yList[Loc==1]
            groupList_year_ = groupList_year[Loc==1]
            groupList_site_ = groupList_site[Loc==1]
            # global_mu = np.mean(yList_)
            
            ## Partially Pooled aka Hierarchical Model
            with pm.Model() as hierarchical_model:
                mu_year = pm.Normal('mu_year', 0, 1) # hyperprior 1
                sigma_year = pm.Exponential('sigma_year', 10) # hyperprior 2, lower lambda, more flatter; higher, more concentrate/small
                       
                mu_site = pm.Normal('mu_site', 0, 1) # hyperprior 1
                sigma_site = pm.Exponential('sigma_site', 10) # hyperprior 2
                
                mu_global = pm.Uniform("phi", lower=0.0, upper=60.0) # hyperprior 1
                sigma_global = pm.Exponential('sigma_global', 10) # hyperprior 2
                
                effect_global = pm.Normal('effect_global', mu_global, sigma_global, shape=1)
                effect_year = pm.Normal('effect_year', mu_year, sigma_year, shape=error.shape[0])
                effect_site = pm.Normal('effect_site', mu_site, sigma_site, shape=error.shape[1])
                
                noise = pm.Exponential('noise', 10)
                
                obs = pm.Normal('obs', effect_global[[0]*len(yList_)] + effect_year[groupList_year_] + effect_site[groupList_site_], noise, observed=yList_)
            
            # fit model
            fig = showStructure(hierarchical_model)
            with hierarchical_model:
                hierarchical_trace = pm.sample(2000, chains=2,
                    return_inferencedata=True,
                    cores=1)
            
            ## global 
            post_global_mu,post_global_sig,percent_97_global,percent_3_global = postStatic(post = hierarchical_trace.posterior.effect_global.values)
            # plt.figure()
            # az.plot_dist(pm.draw(sigma_global,draws=1000))
            plt.figure()
            az.plot_posterior(hierarchical_trace, var_names=['effect_global'])
            # axes = az.plot_forest(hierarchical_trace, var_names=['effect_global'], combined=True)
            
            ## temporal uncertainty
            axes = az.plot_forest(hierarchical_trace, var_names=['effect_year'], combined=True)
            post_temp_mu,post_temp_sig,percent_97_temp,percent_3_temp = postStatic(post = hierarchical_trace.posterior.effect_year.values)
            df_temporal = pd.DataFrame()
            df_temporal['post_temp_mu'] = post_temp_mu
            df_temporal['percent_97'] = percent_97_temp
            df_temporal['percent_3'] = percent_3_temp
            df_temporal['post_temp_sig'] = post_temp_sig
            df_temporal['global_mu'] = post_global_mu[0]
            df_temporal['percent_97_global'] = percent_97_global[0]
            df_temporal['percent_3_global'] = percent_3_global[0]
            df_temporal['post_global_sig'] = post_global_sig[0]
            
            if saveRes:
                df_temporal.to_csv('%s/temporal_%s.csv'%(outPath,crop))
                fig.savefig('%s/structure_%s.png'%(outPath,crop))
                fig2 = axes.ravel()[0].figure
                fig2.savefig('%s/post_temporal_%s.png'%(outPath,crop))
                plt.close(fig)
                plt.close(fig2)
                
            ## site uncertainty
            post_spat_mu,post_spat_sig,percent_97_spat,percent_3_spat = postStatic(post = hierarchical_trace.posterior.effect_site.values)
            df_spatial = pd.DataFrame()
            df_spatial['post_spat_mu'] = post_spat_mu
            df_spatial['percent_97'] = percent_97_spat
            df_spatial['percent_3'] = percent_3_spat
            df_spatial['post_spat_sig'] = post_spat_sig
            df_spatial['global_mu'] = post_global_mu[0]
            df_temporal['percent_97_global'] = percent_97_global[0]
            df_temporal['percent_3_global'] = percent_3_global[0]
            df_temporal['post_global_sig'] = post_global_sig[0]
            # fig post mu
            dataDic = {}
            for s,t in zip(siteList,post_spat_mu):
                dataDic[s] = t
            fig_site_post = visualizeCountyLevel(dataDic,FIPSlist=siteList, crop=crop,
                                      crs = "EPSG:2163", term='Site effect of yield: mu of post',vmin=-20,vmax=25)
            
            # fig post mu + global mu
            dataDic = {}
            for s,t in zip(siteList,post_spat_mu):
                dataDic[s] = t + post_global_mu[0]
            if crop == 'corn':
                fig_site_post2 = visualizeCountyLevel(dataDic,FIPSlist=siteList, crop=crop,
                                          crs = "EPSG:2163", term='Site effect of yield: post mu + global mu',vmin=25,vmax=75)
            else:
                fig_site_post2 = visualizeCountyLevel(dataDic,FIPSlist=siteList, crop=crop,
                                          crs = "EPSG:2163", term='Site effect of yield: post mu + global mu',vmin=0,vmax=45)
            # fig stat mu
            dataDic_prior = {}
            for s,t in zip(siteList,np.nanmean(error-np.nanmean(error),axis=0)):
                dataDic_prior[s] = t
            fig_site_stat = visualizeCountyLevel(dataDic_prior,FIPSlist=siteList, crop=crop,
                                      crs = "EPSG:2163", term='Site effect of yield: mu of stat',vmin=-20,vmax=25)
            # fig post sigma
            dataDic = {}
            for s,t in zip(siteList,post_spat_sig):
                dataDic[s] = t
            fig_site_sig = visualizeCountyLevel(dataDic,FIPSlist=siteList, crop=crop,
                                      crs = "EPSG:2163", term='Site effect of yield: sigma')
            
            if saveRes:
                df_spatial.to_csv('%s/spatial_%s.csv'%(outPath,crop))
                fig_site_post.savefig('%s/site_post_%s.png'%(outPath,crop))
                fig_site_post2.savefig('%s/site_post_mu_plus_globe_%s.png'%(outPath,crop))
                fig_site_stat.savefig('%s/site_stat_%s.png'%(outPath,crop))
                fig_site_sig.savefig('%s/site_sig_%s.png'%(outPath,crop))
                               
                plt.close(fig_site_post)
                plt.close(fig_site_post2)
                plt.close(fig_site_stat)
                plt.close(fig_site_sig)