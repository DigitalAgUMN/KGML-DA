# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 21:02:16 2022

@author: yang8460
"""
import numpy as np
import math
import KGDA_util as util
import os, glob
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['figure.dpi'] = 300

class yieldValidation():
    def __init__(self, yieldPath):
        self.yieldPath = yieldPath
        self.NASSyield()
    
    def coef_C2dryMatter(self, cropType):
        # gC/m2 to ton biomass/m2
        #  dry matter soybean contains 54% carbon, maize contains 45% carbon. g C/m2 to ton dry matter / m2
        if cropType == 0: #Maize
            return 0.01/0.45
        else: # soybean
            return 0.01/0.54
    
    def coef_C2BUacre(self, cropType):
        # gC/m2 to BU/acre
        #  dry matter soybean contains 54% carbon, maize contains 45% carbon.
        wc_corn = 0.156
        wc_soy = 0.13
        conf1 = 0.00220462
        conf2 = 0.000247105
        if cropType == 0: #Maize
            return 1/0.45/(1-wc_corn)*conf1/56/conf2  # 0.4195
        else: # soybean
            return 1/0.54/(1-wc_soy)*conf1/60/conf2   # 0.3165
            
    def NASSyield(self):
        self.yield_NASS_corn_3I = pd.read_csv('%s/3I_2000-2020_corn_organizedYield.csv'%self.yieldPath)
        self.yield_NASS_soybean_3I = pd.read_csv('%s/3I_2000-2020_soybean_organizedYield.csv'%self.yieldPath)
    
    def ecosysYield(self, dataRoot=None, FIPS_dic=None, FIPSList=None):
        out_corn = 'yieldData/yield_ecosys_corn_3I.csv'
        out_soy = 'yieldData/yield_ecosys_soybean_3I.csv'
        if (os.path.exists(out_corn)) & (os.path.exists(out_soy)):
            self.yield_ecosys_corn_3I = pd.read_csv(out_corn)
            self.yield_ecosys_soybean_3I = pd.read_csv(out_soy)
        else:    
            yearRange = np.arange(2000,2020+1)
            yield_ecosys_corn_3I = pd.DataFrame()
            yield_ecosys_corn_3I['Year'] = yearRange
            yield_ecosys_soybean_3I = pd.DataFrame()
            yield_ecosys_soybean_3I['Year'] = yearRange
            
            dic_corn = {}
            dic_soybean = {}
            for n,FIPS in enumerate(FIPSList):
                siteList = FIPS_dic[FIPS]
                countyYield_corn = pd.DataFrame()
                countyYield_soybean = pd.DataFrame()
                countyYield_soybean['Year'] = yearRange
                countyYield_corn['Year'] = yearRange
                
                for site in siteList:                   
                    # load input and Ecosys output data
                    outputData = '%s/%s_outputMerged.pkl'%(dataRoot,site)
                    if not os.path.exists(outputData):
                        print('Site:%s is missing, pass'%site)
                        continue
                    inputData = '%s/%s_inputMerged.pkl'%(dataRoot,site)
                    inputMerged = util.load_object(inputData)
                    outputMerged = util.load_object(outputData)
                    yieldByYear = []
                    cropTypeByYear = []
                    for k,t in zip(inputMerged,outputMerged):
                        yieldByYear.append(t['GrainYield'].tolist()[-1])
                        cropTypeByYear.append(k['CropType'].tolist()[-1])
                    
                    cropTypeByYear_inv = 1-np.array(cropTypeByYear)
                    # separate corn and soybean
                    countyYield_corn[site] = cropTypeByYear_inv*np.array(yieldByYear)
                    countyYield_soybean[site] = np.array(cropTypeByYear)*np.array(yieldByYear)
                     
                tmp = np.array(countyYield_corn)[:,1:]
                if tmp.shape[1]>0:
                    tmp[tmp==0] = np.nan
                    # yield_ecosys_corn_3I[FIPS] = np.nanmean(tmp,axis=1)
                    dic_corn[FIPS] = np.nanmean(tmp,axis=1)
                    
                tmp = np.array(countyYield_soybean)[:,1:]
                if tmp.shape[1]>0:
                    tmp[tmp==0] = np.nan
                    # yield_ecosys_soybean_3I[FIPS] = np.nanmean(tmp,axis=1)
                    dic_soybean[FIPS] = np.nanmean(tmp,axis=1)
                    
                print('%d/%d counties finished.'%(n+1,len(FIPSList)))
            self.yield_ecosys_corn_3I = pd.concat([yield_ecosys_corn_3I,pd.DataFrame(dic_corn)],axis=1)
            self.yield_ecosys_soybean_3I = pd.concat([yield_ecosys_soybean_3I,pd.DataFrame(dic_soybean)],axis=1)
            self.yield_ecosys_corn_3I.to_csv('yieldData/yield_ecosys_corn_3I.csv')
            self.yield_ecosys_soybean_3I.to_csv('yieldData/yield_ecosys_soybean_3I.csv')


def NASS_vs_ecosys(NASS_Path):
    y = yieldValidation(NASS_Path)
    y.ecosysYield()
    year = y.yield_NASS_corn_3I['Year'].tolist()
    
    # corn
    NASS_corn = np.nanmean(np.array(y.yield_NASS_corn_3I)[:,2:],axis=1)
    ecosys_corn = np.nanmean(np.array(y.yield_ecosys_corn_3I)[:,2:],axis=1)* y.coef_C2BUacre(0)
    plt.figure(figsize=(10,5))
    plt.plot(year,NASS_corn,'r-',label='NASS_corn')
    plt.plot(year,ecosys_corn,'g-',label='ecosys_corn')
    
    # soybean
    NASS_soybean = np.nanmean(np.array(y.yield_NASS_soybean_3I)[:,2:],axis=1)
    ecosys_soybean = np.nanmean(np.array(y.yield_ecosys_soybean_3I)[:,2:],axis=1)* y.coef_C2BUacre(1)
    plt.figure(figsize=(10,5))
    plt.plot(year,NASS_soybean,'r-',label='NASS_soybean')
    plt.plot(year,ecosys_soybean,'g-',label='ecosys_soybean')
    plt.legend()
    
    # scatter plot
    scatterYield(df_NASS = y.yield_NASS_corn_3I, df_p = y.yield_ecosys_corn_3I, 
                 coef = y.coef_C2BUacre(0), title='Ecosys vs. NASS corn')
    scatterYield(df_NASS = y.yield_NASS_soybean_3I, df_p = y.yield_ecosys_soybean_3I, 
                 coef = y.coef_C2BUacre(1), title='Ecosys vs. NASS soybean')
    
def scatterYield(df_NASS,df_p, coef = 1, title='',saveFig=False,outFolder=None,note=''):
    # cal intersection
    validFIPS = list(set(df_NASS.columns[2:]).intersection(set(df_p.columns[2:])))
    validFIPS.sort()
    
    # obs and pre
    obs = []
    pre = []
    obs_yearMean = []
    pre_yearMean = []
    dic_obs = {}
    dic_pre = {}
    YearList = df_NASS['Year'].tolist()
    dic_obs['Year'] = YearList
    dic_pre['Year'] = YearList
    
    for FIPS in validFIPS:
        obs.extend(df_NASS[FIPS].tolist())
        pre.extend(df_p[FIPS].tolist())
        t1,t2 = removeNaNmean(df_NASS[FIPS],df_p[FIPS])
        obs_yearMean.append(t1)
        pre_yearMean.append(t2)
        dic_obs[FIPS] = df_NASS[FIPS]/coef
        dic_pre[FIPS] = df_p[FIPS]
    
    dic_obs = pd.DataFrame(dic_obs)
    dic_pre = pd.DataFrame(dic_pre)
    
    difference = dic_pre-dic_obs
    difference['Year'] = YearList
    x_,y_ = removeNaN(obs,pre,coef=coef) 
    util.plotScatterDense(x_=x_, y_=y_, binN=100 ,title=title,saveFig=saveFig,outFolder=outFolder,note='all_%s'%note)

    x_,y_ = removeNaN(obs_yearMean,pre_yearMean,coef=coef) 
    util.plotScatterDense(x_=x_, y_=y_, binN=100, title='Multi year mean %s'%title,
                          saveFig=saveFig,outFolder=outFolder,note='multiYearAve_%s'%note)
    
    summary = {}
    summary['diff'] = difference
    summary['obs'] = dic_obs
    summary['pre'] = dic_pre
    return summary

def removeNaN(obs,pre,coef=1):
    x = np.array(obs)
    y = np.array(pre)*coef
    
    Loc = (1 - (np.isnan(x) | np.isnan(y)))
    x_ = x[Loc==1]
    y_ = y[Loc==1]
    return x_,y_

def removeNaNmean(obs,pre,coef=1):
    x = np.array(obs)
    y = np.array(pre)*coef
    
    Loc = (1 - (np.isnan(x) | np.isnan(y)))
    x[Loc==0] = np.nan
    y[Loc==0] = np.nan
    if len(x.shape)==2:
        return np.nanmean(x,axis=1),np.nanmean(y,axis=1)
    else:
        return np.nanmean(x),np.nanmean(y)

def NASS_vs_EcoNet(NASS_Path,NetResult,saveFig=False,outFolder=None):
    y = yieldValidation(NASS_Path)
    year = y.yield_NASS_corn_3I['Year'].tolist()
    
    # corn   
    yield_ecoNet_corn_3I = pd.read_csv('%s/yield_3I_corn.csv'%NetResult)
    validFIPS = list(set(y.yield_NASS_corn_3I.columns[2:]).intersection(set(yield_ecoNet_corn_3I.columns[2:])))
    validFIPS.sort()
    
    NASS_corn = np.array(y.yield_NASS_corn_3I[validFIPS])[:-1,:]
    ecoNet_corn = np.array(yield_ecoNet_corn_3I[validFIPS])* y.coef_C2BUacre(0)
    NASS_corn,ecoNet_corn=removeNaNmean(NASS_corn,ecoNet_corn)
    fig = plt.figure(figsize=(10,5))
    plt.plot(year[:-1],NASS_corn,'r-',label='NASS_corn')
    plt.plot(year[:-1],ecoNet_corn,'g--',label='ecoNet_corn')
    plt.legend()
    plt.xlabel('Year')
    plt.ylabel('Yield BU/acre')
    if saveFig:
        plt.title('Yield trend corn')
        fig.savefig('%s/yieldTrendCorn.png'%(outFolder))
        
    # soybean
    try:
        yield_ecoNet_soybean_3I = pd.read_csv('%s/yield_3I_soybean.csv'%NetResult)
    except:
        yield_ecoNet_soybean_3I = pd.read_csv('%s/yield_3I_soy.csv'%NetResult)

     
    validFIPS = list(set(y.yield_NASS_soybean_3I.columns[2:]).intersection(set(yield_ecoNet_soybean_3I.columns[2:])))
    validFIPS.sort()
    NASS_soybean = np.array(y.yield_NASS_soybean_3I[validFIPS])[:-1,:]
    ecoNet_soybean = np.array(yield_ecoNet_soybean_3I[validFIPS])* y.coef_C2BUacre(1)
    NASS_soybean,ecoNet_soybean=removeNaNmean(NASS_soybean,ecoNet_soybean)
    fig = plt.figure(figsize=(10,5))
    plt.plot(year[:-1],NASS_soybean,'r-',label='NASS_soybean')
    plt.plot(year[:-1],ecoNet_soybean,'g--',label='ecoNet_soybean')
    plt.legend()
    plt.xlabel('Year')
    plt.ylabel('Yield BU/acre')
    if saveFig:
        plt.title('Yield trend soybean')
        fig.savefig('%s/yieldTrendSoybean.png'%(outFolder))
        
    # scatter plot
    summary_corn = scatterYield(df_NASS = y.yield_NASS_corn_3I.iloc[:-1], df_p = yield_ecoNet_corn_3I, 
                 coef = y.coef_C2BUacre(0), title='EcoNet vs. NASS corn',
                 saveFig=saveFig,outFolder=outFolder,note='corn')
    summary_soybean = scatterYield(df_NASS = y.yield_NASS_soybean_3I.iloc[:-1], df_p = yield_ecoNet_soybean_3I, 
                 coef = y.coef_C2BUacre(1), title='EcoNet vs. NASS soybean',
                 saveFig=saveFig,outFolder=outFolder,note='soybean')

    return summary_corn, summary_soybean

def NASS_vs_EcoNet_cornBelt(NASS_Path,NetResult,saveFig=False,outFolder=None):
    y = util.yieldValidation(NASS_Path)
    year = y.yield_NASS_corn['Year'].tolist()
    
    # corn   
    yield_ecoNet_corn = pd.read_csv('%s/yield_corn.csv'%NetResult)
    validFIPS = list(set(y.yield_NASS_corn.columns[2:]).intersection(set(yield_ecoNet_corn.columns[2:])))
    validFIPS.sort()
    
    NASS_corn = np.array(y.yield_NASS_corn[validFIPS])
    ecoNet_corn = np.array(yield_ecoNet_corn[validFIPS])* y.coef_C2BUacre(0)
    NASS_corn,ecoNet_corn=removeNaNmean(NASS_corn,ecoNet_corn)
    fig = plt.figure(figsize=(10,5))
    plt.plot(year,NASS_corn,'r-',label='NASS_corn')
    plt.plot(year,ecoNet_corn,'g--',label='ecoNet_corn')
    plt.legend()
    plt.xlabel('Year')
    plt.ylabel('Yield BU/acre')
    if saveFig:
        plt.title('Yield trend corn')
        fig.savefig('%s/yieldTrendCorn.png'%(outFolder))
        
    # soybean
    try:
        yield_ecoNet_soybean = pd.read_csv('%s/yield_soybean.csv'%NetResult)
    except:
        yield_ecoNet_soybean = pd.read_csv('%s/yield_soy.csv'%NetResult)

     
    validFIPS = list(set(y.yield_NASS_soybean.columns[2:]).intersection(set(yield_ecoNet_soybean.columns[2:])))
    validFIPS.sort()
    NASS_soybean = np.array(y.yield_NASS_soybean[validFIPS])
    ecoNet_soybean = np.array(yield_ecoNet_soybean[validFIPS])* y.coef_C2BUacre(1)
    NASS_soybean,ecoNet_soybean=removeNaNmean(NASS_soybean,ecoNet_soybean)
    fig = plt.figure(figsize=(10,5))
    plt.plot(year,NASS_soybean,'r-',label='NASS_soybean')
    plt.plot(year,ecoNet_soybean,'g--',label='ecoNet_soybean')
    plt.legend()
    plt.xlabel('Year')
    plt.ylabel('Yield BU/acre')
    if saveFig:
        plt.title('Yield trend soybean')
        fig.savefig('%s/yieldTrendSoybean.png'%(outFolder))
        
    # scatter plot
    summary_corn = scatterYield(df_NASS = y.yield_NASS_corn, df_p = yield_ecoNet_corn, 
                 coef = y.coef_C2BUacre(0), title='EcoNet vs. NASS corn',
                 saveFig=saveFig,outFolder=outFolder,note='corn')
    summary_soybean = scatterYield(df_NASS = y.yield_NASS_soybean, df_p = yield_ecoNet_soybean, 
                 coef = y.coef_C2BUacre(1), title='EcoNet vs. NASS soybean',
                 saveFig=saveFig,outFolder=outFolder,note='soybean')

    return summary_corn, summary_soybean

if __name__ == '__main__':
    # load county-level sites
    NASS_Path = 'E:/NASS_yield'

    # NASS_vs_ecosys
    # NASS_vs_ecosys(NASS_Path)
    
    # NASS vs econet
    # NetResult = 'yieldResult/county_yield-220730-220933'
    # NetResult = 'yieldResult/county_yield_openloop-220801-000629'
    # NetResult = 'yieldResult/county_yield_parallel-220807-230023'
    # NetResult = 'yieldResult/county_yield_parallel-220809-162403'
    # NetResult = 'yieldResult/county_yield_parallel_25296-220810-015851'
    # NetResult = 'yieldResult/county_yield_parallel_3layerDecoderParaPheno_c2_maskYield_25296-220821-114843'
    # NetResult = 'yieldResult/county_yield_parallel_3layerDecoderParaPheno_c2_maskNone_25296-220821-174625'
    # NetResult = 'yieldResult/county_yield_parallel_3layerDecoderParaPheno_c2_maskPheno_25296-220822-093415'
    # NetResult = 'yieldResult/county_yield_parallel_openloop_25296-220810-235926'
    # NetResult = 'yieldResult/county_yield_parallel_batchsite-220829-174105'
    # NetResult = 'yieldResult/county_yield_parallel_3layerDecoderParaPheno_c2_maskc3c4_dailyObs_25296-220826-002513'
    # NetResult = 'yieldResult/county_yield_parallel_3layerDecoderParaPheno_c2_maskc3c4_dailyObs_25296-220901-154749'
    NetResult = 'F:\OneDrive - whu.edu.cn\ecosys_RNN\paraCaliResults\PSO\case12'
    # NetResult = 'yieldResult/county_yield_parallel_epoch60_3layerDecoderParaPheno_c2_maskc3c4_obsInterval10_countyMerge-220906-113541'
    summary_corn, summary_soybean=NASS_vs_EcoNet(NASS_Path,NetResult)