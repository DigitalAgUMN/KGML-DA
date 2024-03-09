# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 14:41:08 2022

@author: yang8460

"""
import numpy as np
import KGDA_util as util
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from scipy import stats

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['figure.dpi'] = 300

def correctTrend(df,para,midYear,coef=1):
    yearList = df.Year.values
    yearDiff = yearList - midYear
    offset = yearDiff*para[0]
    keys = df.columns.tolist()[1:]
    df_np = np.array(df)[:,1:]*coef
    df_np += np.tile(offset,(len(keys),1)).T
    df_new = pd.DataFrame(df_np,columns=keys)
    df_new.insert(0,'Year',yearList)
    
    return df_new

def correctTrend_indiv(df,paraDic,midYear,coef=1,validFIPS=None):
    yearList = df.Year.values
    df_new = {}
    df_new['Year'] = yearList
    for t in validFIPS:        
        yearDiff = yearList - midYear
        offset = yearDiff*paraDic[t][0]
        df_new[t] = np.array(df[t])*coef + offset
           
    return pd.DataFrame(df_new)

def globalTrend(df_yield_nass,outFolder,crop,mode,interval,yearRange_t,coef):
    # calculate the trend of NASS yield
    trend = np.nanmean(np.array(df_yield_nass)[:,1:],axis=1)
    trendYear = df_yield_nass.Year.values 
    para = np.polyfit(trendYear, trend, 1)
    y_fit = np.polyval(para, trendYear)  #
    plt.figure(figsize = (12,5))
    plt.plot(trendYear, y_fit, 'r')
    plt.plot(trendYear, trend, 'b.')
    delta_y = para[0]
    
    # correct predicted yield trend    
    yield_df = pd.read_csv('%s/yield_%s.csv'%(outFolder,crop))
    yield_df.drop(['Unnamed: 0'],inplace=True,axis=1)
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
                tmp = np.array(yield_df[yield_df['Year']==year])[:,1:]*coef+(year-midYear)*delta_y
            
            recorrected.append(tmp)
    keys = yield_df.columns.tolist()[1:]
    df_new = pd.DataFrame(np.concatenate(recorrected,axis=0),columns=keys)
    df_new.insert(0,'Year',yearRange_t)        
    
    return df_new, yield_df

def individualTrend(df_yield_nass,outFolder,crop,mode,interval,yearRange_t,coef=1):
    # calculate the trend of NASS yield
    yield_df = pd.read_csv('%s/yield_%s.csv'%(outFolder,crop))
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

def plotScatterDense(x_, y_,alpha=1, binN=200, thresh_p = 0.05,outFolder='',
                     saveFig=False,note='',title='',uplim=None,downlim=None,
                     auxText = None,legendLoc=4, cmap='Reds', removeNeg=False,
                     vmin=None,vmax=None,removeZero=True,upcoef=1.2,i=0,crop='corn'):
    
    ax = plt.subplot(2, 4, i+1)
    if crop=='corn':
        ax.set_facecolor('y')
    else:
        ax.set_facecolor('g')
    ax.patch.set_alpha(0.2)
    
    x_=np.array(x_)
    y_=np.array(y_)
    if len(y_) > 1:
        if removeZero:
            loc = ((x_!=0) & (y_!=0))
            x_ = x_[loc]
            y_ = y_[loc]
            
        if removeNeg:
            loc = ((x_>0) & (y_>0))
            x_ = x_[loc]
            y_ = y_[loc]
        # Calculate the point density
        if not (thresh_p is None):
            thresh = (np.max(np.abs(x_))*thresh_p)
            loc = ((x_>thresh)|(x_<-thresh)) & ((y_>thresh)|(y_<-thresh))
            x_ = x_[loc]
            y_ = y_[loc]

        x=x_
        y=y_
        tmp = stats.linregress(x, y)
        para = [tmp[0],tmp[1]]
        # para = np.polyfit(x, y, 1)   # can't converge for large dataset
        y_fit = np.polyval(para, x)  #
        # plt.plot(x, y_fit, 'r')
    
    #histogram definition
    bins = [binN, binN] # number of bins
    
    # histogram the data
    hh, locx, locy = np.histogram2d(x, y, bins=bins)

    # Sort the points by density, so that the densest points are plotted last
    z = np.array([hh[np.argmax(a<=locx[1:]),np.argmax(b<=locy[1:])] for a,b in zip(x,y)])
    idx = z.argsort()
    x2, y2, z2 = x[idx], y[idx], z[idx]

    # plt.scatter(x2, y2, c=z2, cmap=cmap, marker='.',alpha=alpha)
    plt.scatter(x2, y2, c=z2, cmap=cmap, marker='.',alpha=alpha,vmin=vmin,vmax=vmax)
    if uplim==None:
        uplim = upcoef*max(np.hstack((x, y)))
    if downlim==None:
        if i%2 == 0:
            downlim = 0
            auxText = numberList[i]
            title = 'All county-year'
        else:
            downlim = 0.8*min(np.hstack((x, y)))
            auxText = numberList[i]  
            title = 'Multi-year average'
    figRange = uplim - downlim
    plt.plot(np.arange(downlim-1,np.ceil(uplim)+1), np.arange(downlim-1,np.ceil(uplim)+1), 'k', label='1:1 line')
    plt.xlim([downlim, uplim])
    plt.ylim([downlim, uplim])

    if not legendLoc is None:
        if legendLoc==False:
            plt.legend(edgecolor = 'w',facecolor='w',fontsize=12)          
        else:
            plt.legend(loc = legendLoc, edgecolor = 'w',facecolor='w',fontsize=12, framealpha=0)
    plt.title(title, y=1, fontsize=20)
    
    if len(y) > 1:
        R2 = np.corrcoef(x, y)[0, 1] ** 2
        RMSE = (np.sum((y - x) ** 2) / len(y)) ** 0.5
        Bias = np.mean(y) - np.mean(x)
        MAE = np.sum(np.abs(y - x)) / len(y)
        # MAPE = 100 * np.sum(np.abs((y - x) / (x+0.00001))) / len(x)
        plt.text(downlim + 0.05 * figRange, downlim + 0.90 * figRange, r'$R^2 $= ' + str(R2)[:5], fontsize=14)
        # ax = plt.text(0.1 * uplim, 0.86 * uplim, r'$MAPE $= ' + str(MAPE)[:5] + '%', fontsize=14)
        plt.text(downlim + 0.05 * figRange, downlim + 0.83 * figRange, r'$RMSE $= ' + str(RMSE)[:5], fontsize=14)
    
        # plt.text(downlim + 0.1 * figRange, downlim + 0.69 * figRange, r'$Slope $= ' + str(para[0])[:5], fontsize=14)
        plt.text(downlim + 0.05 * figRange, downlim + 0.76 * figRange, r'$Bias $= ' + str(Bias)[:5], fontsize=14)
        plt.text(downlim + 0.05 * figRange, downlim + 0.69 * figRange, r'$MAE $= ' + str(MAE)[:5], fontsize=14)
        
    if not auxText == None:
        plt.text(-0.01, 1.03, auxText, transform=ax.transAxes,fontproperties = 'Times New Roman',fontsize=20)
    if crop == 'corn':    
        plt.text(0.7, 0.15, 'Corn', transform=ax.transAxes,fontproperties = 'Times New Roman',fontsize=20)
    else:
        plt.text(0.65, 0.15, 'Soybean', transform=ax.transAxes,fontproperties = 'Times New Roman',fontsize=20)
    plt.colorbar()
    plt.tight_layout()

def plotScatter(x_, y_,thresh_p = 0.05,outFolder='',
                     saveFig=False,note='',title='',uplim=None,downlim=None,
                     auxText = None,legendLoc=4,  removeNeg=False,
                    removeZero=True,upcoef=1.2,i=0,crop='corn'):
        
    ax = plt.subplot(2, 4, i+1)
    if crop=='corn':
        ax.set_facecolor('y')
    else:
        ax.set_facecolor('g')
    ax.patch.set_alpha(0.2)
    
    x_=np.array(x_)
    y_=np.array(y_)
    if len(y_) > 1:
        if removeZero:
            loc = ((x_!=0) & (y_!=0))
            x_ = x_[loc]
            y_ = y_[loc]
            
        if removeNeg:
            loc = ((x_>0) & (y_>0))
            x_ = x_[loc]
            y_ = y_[loc]
        # Calculate the point density
        if not (thresh_p is None):
            thresh = (np.max(np.abs(x_))*thresh_p)
            loc = ((x_>thresh)|(x_<-thresh)) & ((y_>thresh)|(y_<-thresh))
            x_ = x_[loc]
            y_ = y_[loc]

        x=x_
        y=y_
        tmp = stats.linregress(x, y)
        para = [tmp[0],tmp[1]]
        # para = np.polyfit(x, y, 1)   # can't converge for large dataset
        y_fit = np.polyval(para, x)  #
        # plt.plot(x, y_fit, 'r')
    

    plt.scatter(x, y, marker='.', color = 'k')
    if uplim==None:
        uplim = upcoef*max(np.hstack((x, y)))
    if downlim==None:
        downlim = 0.8*min(np.hstack((x, y)))
        
    figRange = uplim - downlim
    plt.plot(np.arange(downlim-1,np.ceil(uplim)+1), np.arange(downlim-1,np.ceil(uplim)+1), 'k', label='1:1 line')
    plt.xlim([downlim, uplim])
    plt.ylim([downlim, uplim])
    # plt.xlabel('Observed yield (Bu/Acre)',fontsize=16)
    # plt.ylabel('Predicted yield (Bu/Acre)',fontsize=16)
    # plt.legend(loc=1)  # 指定legend的位置,读者可以自己help它的用法
    if not legendLoc is None:
        if legendLoc==False:
            plt.legend(edgecolor = 'w',facecolor='w',fontsize=12)          
        else:
            plt.legend(loc = legendLoc, edgecolor = 'w',facecolor='w',fontsize=12, framealpha=0)
    plt.title(title, y=0.9, fontsize=14)
    
    if len(y) > 1:
        R2 = np.corrcoef(x, y)[0, 1] ** 2
        RMSE = (np.sum((y - x) ** 2) / len(y)) ** 0.5
        MAE = np.sum(np.abs(y - x)) / len(y)
        Bias = np.mean(y) - np.mean(x)
        # MAPE = 100 * np.sum(np.abs((y - x) / (x+0.00001))) / len(x)
        plt.text(downlim + 0.1 * figRange, downlim + 0.83 * figRange, r'$R^2 $= ' + str(R2)[:5], fontsize=14)
        # ax = plt.text(0.1 * uplim, 0.86 * uplim, r'$MAPE $= ' + str(MAPE)[:5] + '%', fontsize=14)
        plt.text(downlim + 0.1 * figRange, downlim + 0.76 * figRange, r'$RMSE $= ' + str(RMSE)[:5], fontsize=14)
    
        # plt.text(downlim + 0.1 * figRange, downlim + 0.69 * figRange, r'$Slope $= ' + str(para[0])[:5], fontsize=14)
        plt.text(downlim + 0.1 * figRange, downlim + 0.69 * figRange, r'$Bias $= ' + str(Bias)[:5], fontsize=14)
        plt.text(downlim + 0.1 * figRange, downlim + 0.62 * figRange, r'$MAE $= ' + str(MAE)[:5], fontsize=14)
        
    if not auxText == None:
        plt.text(0.05, 0.9, auxText, transform=ax.transAxes,fontproperties = 'Times New Roman',fontsize=20)
    plt.colorbar()
    plt.tight_layout()    
        
def scatterYield(df_NASS,df_p, coef = 1, title='',saveFig=False,outFolder=None,note='',aveYear=True,n=0,crop='corn'):
    # cal intersection
    validFIPS = list(set(df_NASS.columns[1:]).intersection(set(df_p.columns[1:])))
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
    
    obs_siteMean = np.nanmean(df_NASS[validFIPS].values,axis=1)
    pre_siteMean = np.nanmean(df_p[validFIPS].values,axis=1)

    for FIPS in validFIPS:        
        obs.extend(df_NASS[FIPS].tolist())
        pre.extend(df_p[FIPS].tolist())
        t1,t2 = util.removeNaNmean(df_NASS[FIPS],df_p[FIPS])
        obs_yearMean.append(t1)
        pre_yearMean.append(t2)

        dic_obs[FIPS] = df_NASS[FIPS]/coef
        dic_pre[FIPS] = df_p[FIPS]
    
    dic_obs = pd.DataFrame(dic_obs)
    dic_pre = pd.DataFrame(dic_pre)
    
    difference = dic_pre.reset_index(drop=True)-dic_obs.reset_index(drop=True)
    difference['Year'] = YearList
    x_,y_ = util.removeNaN(obs,pre,coef=coef)
    
    vmin=1
    vmax=30
    upcoef = 1.2
    number = ''
    plotScatterDense(x_=x_, y_=y_, binN=100 ,title='',cmap='jet',saveFig=saveFig, 
                     outFolder=outFolder,vmin=vmin, vmax=vmax,upcoef=upcoef,i=n,crop=crop)

    vmin=1
    vmax=4
    upcoef = 1.2
    number = ''
    if aveYear:
        x_,y_ = util.removeNaN(obs_yearMean,pre_yearMean,coef=coef) 
        plotScatterDense(x_=x_, y_=y_, binN=100, title='',cmap='jet',
                              auxText=number, outFolder=outFolder, saveFig=saveFig,
                         vmin=vmin, vmax=vmax,upcoef=upcoef,i=n+1,crop=crop)
    else:
        plotScatter(x_=obs_siteMean, y_=pre_siteMean, title='',
                              auxText=number, outFolder=outFolder, saveFig=saveFig,
                        upcoef=upcoef,i=n+1,crop=crop)
        
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
    
    return modeDic, outFolderDic

if __name__ == '__main__':

    saveFig = False#True
    globalT = False#True#
    interval=3   
    yearRange_t = [t for t in range(2000,2020+1)]
    
    # the path of results, you can change it to your own pathes
    modeDic, outFolderDic = casePathes()
    
    numberList = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)']
    fig = plt.figure(figsize=(16,7))
    n=0
    for case in ['default_op','default_GPP']:
        outFolder = outFolderDic[case]
        mode = modeDic[case]
        for crop in ['corn','soybean']:
            print('Processing %s'%(crop))
            
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
                  
            # exteact the NASS yield
            loc = [i for i,t in enumerate(df_yield_nass.Year.tolist()) if t in yearRange_t]
            df_yield_nass_ = df_yield_nass.iloc[loc]
            if globalT:
                yield_df, yield_df_origin = globalTrend(df_yield_nass_,outFolder,crop,mode,interval,yearRange_t,coef=coef)
            else:
                yield_df, yield_df_origin = individualTrend(df_yield_nass_,outFolder,crop,mode,interval,yearRange_t,coef=coef)
            
            validFIPS_GPP,_ = util.validGPPcounty()
            validFIPS_GPP_ = list(set(validFIPS_GPP).intersection(set(yield_df.columns)))
            
            # show trend corrected discard first three years and some nan couties
            df_NASS=df_yield_nass_.iloc[3:]
            df_p=yield_df[['Year']+validFIPS_GPP_].iloc[3:]
            
            year = df_NASS['Year'].tolist()
            validFIPS = list(set(df_NASS.columns[1:]).intersection(set(df_p.columns[1:])))
            validFIPS.sort()
    
            # scatter plot    
            scatterYield(df_NASS, df_p, title='KGML-DA vs. NASS, %s yield (BU/acre) %s'%(crop,mode),
                         saveFig=saveFig,outFolder=outFolder,note = '%s_%s'%(crop,mode),n=n,crop=crop)
     
            n+=2
    fig.text(0.5, -0.015, 'Observed yield (Bu/Acre)', ha='center',fontsize=22)
    fig.text(-0.01, 0.75, 'Open-loop', va='center', rotation='vertical',fontsize=20)
    fig.text(-0.01, 0.28, 't-EnKF', va='center', rotation='vertical',fontsize=20)
    fig.text(-0.035, 0.5, 'Predicted yield (Bu/Acre)', va='center', rotation='vertical',fontsize=22)