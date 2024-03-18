# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 16:47:47 2022

@author: yang8460

case11, cali year same with the test year, e.g., 2000 cali, 2000 test
v2, set scenarios for calibration set
"""

import numpy as np
import KGDA_util as util
import matplotlib.pyplot as plt
import KGDA_Networks as net
import torch
import datetime
import copy
import os, glob
import scipy.signal
import pandas as pd
# from pymoo.core.problem import ElementwiseProblem
import time
import matplotlib
import ECONET_Test_county_plot as plot
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['figure.dpi'] = 300

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device}" " is available.")

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class makeInput():  
    def __init__(self, inputMerged,paraMode,startYear=2000):
        
        self.inputMerged=inputMerged
        self.startYear = startYear
        
        ## in & out
        self.cropParaList = ['VCMX','CHL4','GROUPX','STMX','GFILL','SLA1']
        self.cropPara_index = [7,8,9,10,11,12]
        if paraMode== 'defaultV2':
            self.cropParaDefaults = [[90, 0.05, 17, 5, 0.0005, 0.017],
                            [45, 0.0, 18, 4, 0.0005, 0.011]]  # 0:maize,  1:soybean
        else:               
            self.cropParaDefaults = [[90, 0.05, 19, 5, 0.0005, 0.018],
                                [45, 0.0, 17, 4, 0.0005, 0.01]]  # 0:maize,  1:soybean
        
        
        self.y_selectFeatures = ['DVS',
                            'ET_daily','GPP_daily','AVE_SM',
                            'Biomass','Reco_daily','NEE_daily',
                            'LAI','GrainYield']
        self.y_NormCoef = [0.5,
                      0.15, 0.02, 1.5,
                      0.001, 0.06, -0.05,
                      0.1,0.0015]
        self.fert = 5
        
        self.X_selectFeatures = ['Tair','RH','Wind','Precipitation','Radiation','GrowingSeason',
                            'CropType','VCMX','CHL4','GROUPX','STMX','GFILL','SLA1','Bulk density','Field capacity'
                            ,'Wilting point','Ks','Sand content','Silt content','SOC','Fertilizer']   
        self.plantingDate_corn = [5,1] #  planting date
        self.plantingDate_soybean = [6,1]
        self.DisturbDays = 15  # disturb +_ days
        
    def get(self, year):
        self.year = year
        yearList = [t for t in range(self.startYear, self.startYear+len(self.inputMerged[0]))]
        loc = yearList.index(year)
        out = []
        for t in self.inputMerged:
            tmp = t[loc]
            if tmp is not None:
                out.append(tmp[self.X_selectFeatures])
            else:
                out.append(None)
        return out
    
    def resetDefaultPara(self,inputEpisodes,cropTypes,cropParaCali=None):
        inputEpisode_reset_list = []
        n=0
        for inputEpisode,cropType in zip(inputEpisodes,cropTypes):
            if inputEpisode is None:
                inputEpisode_reset_list.append(None)
            else:
                inputEpisode_reset = inputEpisode.copy()
                inputEpisode_reset['CropType'] = cropType
                
                if cropType==0:
                    inputEpisode_reset['Fertilizer'] = self.fert
                    dateP = datetime.date(self.year,self.plantingDate_corn[0],self.plantingDate_corn[1])
                else:
                    inputEpisode_reset['Fertilizer'] = 0
                    dateP = datetime.date(self.year,self.plantingDate_soybean[0],self.plantingDate_soybean[1])
                          
                for v, vi in zip(self.cropParaDefaults[cropType],self.cropParaList):
                        inputEpisode_reset[vi] = v
                                         
                # additional replace the parameters from calibration results
                if cropParaCali is not None:
                    paraList = [self.X_selectFeatures[t] for t in cropParaCali[1]]
                    if not np.isnan(cropParaCali[0][n,0]):                        
                        for v, vi in zip(cropParaCali[0][n,:],paraList):
                            inputEpisode_reset[vi] = v
                        
                # # set growing season
                Pd = round(inputEpisode_reset['GrowingSeason'].values[0])
                if Pd == 0:
                    dateP_DOY = dateP.timetuple().tm_yday
                else:
                    dateP_DOY = Pd
                tmp = np.zeros(len(inputEpisode_reset))
                tmp[dateP_DOY-1:] = 1
                inputEpisode_reset['GrowingSeason'] = tmp
                inputEpisode_reset_list.append(inputEpisode_reset)
            n+=1
        return inputEpisode_reset_list
    
    def validCheck(self,inputEpisodes,FIPSbatch):
        validFIPS = []
        inputEpisodes_vali = []
 
        for inputEpisode,FIPS in zip(inputEpisodes,FIPSbatch):
            if inputEpisode is None:
                continue
            validFIPS.append(FIPS)
            inputEpisode=np.array(inputEpisode).astype(np.float32)
            inputEpisodes_vali.append(inputEpisode)
        return  inputEpisodes_vali,validFIPS
        
class ECONET():
    def __init__(self):
        input_dim = 21
        output_dim=[1,3,3,2]
        hidden_dim = 64
        mode='paraPheno_c2'
        self.model = to_device(net.KG_ecosys(input_dim=input_dim,hidden_dim=hidden_dim,
                                                    output_dim=output_dim,mode=mode), device)
        modelName ='model/gru-epoch30-batch256-cornBelt20299_4cell_v1_2_RecoCorrected_paraPheno_c2-221010-000709_state_dict.pth'
        self.model.load_state_dict(torch.load(modelName))
        
    def run(self,x):
        self.model.eval()
        out,_ = self.model(torch.tensor(x.astype(np.float32)).to(device))
        return out.detach().cpu().numpy()
    
def replaceInpput(inputDataList,xList,paraLoc):
        inputSample_en = []
        for inputData,xs in zip(inputDataList,xList):
            length = inputData.shape[0]
            para = np.array([t for t in xs]*length).reshape(length,-1)
            # planting data
            DOY_p = int(xs[0])
            tmp = np.zeros((length))
            tmp[DOY_p-1:,] = 1
            para[:,0] = tmp
            inputSample = inputData.copy()
            inputSample[:,paraLoc] = para
            inputSample_en.append(inputSample)
            
        return inputSample_en

def scatterYield(df_NASS,df_p, coef = 1, title='',saveFig=False,outFolder=None,note=''):
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
    
def NASS_vs_ecosys(df_NASS,df_p,crop,coef,outFolder='',mode='',saveFig=False):
    year = df_NASS['Year'].tolist()
    validFIPS = list(set(df_NASS.columns[1:]).intersection(set(df_p.columns[1:])))
    validFIPS.sort()
    
    # corn
    NASS_ =np.array(df_NASS[validFIPS])[:,1:]
    ecoNet_ = np.array(df_p[validFIPS])[:,1:]* coef
    NASS,ecoNet=removeNaNmean(NASS_,ecoNet_)
    
    fig = plt.figure(figsize=(12,5))
    plt.plot(year,NASS,'r-',marker='.',label='NASS_corn')
    plt.plot(year,ecoNet,'g-',marker='*',label='ecosys_corn')
    plt.ylabel('Yield BU/Acre',fontsize=14)
    plt.xlabel('Year',fontsize=14)
    
    if saveFig:
        fig.savefig('%s/yield_3I_trend_%s_%s.png'%(outFolder,crop,mode))
    # scatter plot
    summary = scatterYield(df_NASS, df_p, 
                 coef = coef, title='Ecosys vs. NASS %s BU/acre'%crop,
                 saveFig=saveFig,outFolder=outFolder,note = '%s_%s'%(crop,mode))
    return  summary

def subplotTrend(target_dic,yearAxis,varList,crop = 'corn',item=''):                        
    fig=plt.figure(figsize=(12,11))
    
    merge = np.stack([np.array(t) for t in target_dic.values()]).transpose((1,0,2))
    for i,p in enumerate(varList):
        ax = plt.subplot(3,3,i+1)
        ymean = np.mean(merge,axis=1)[:,i]
        plt.plot(yearAxis, ymean,'k--')
        ystd = np.std(merge,axis=1)[:,i]
        plt.ylabel(p)
        ax.fill_between(yearAxis, ymean- ystd,ymean + ystd, alpha=0.2)
        
    plt.suptitle('Vars %s trend of %s'%(item,crop),y=0.9)
    
    return fig
if __name__ == '__main__':
    model = ECONET()
    
    # input data
    # setting
    saveRes =True
    batchFIPS = 320
    version = ''#'_v2'
    dataRoot = 'F:/MidWest_counties/inputMerged_DA_countyMerge'
    GPPpath = 'F:/MidWest_counties/GPP'
    countyPathes = glob.glob('%s/*.pkl'%dataRoot)
    FIPSList_all = [t.split('\\')[-1].split('_')[0] for t in countyPathes]
    
    algorithm = 'PSO'#'SCEUA'
    yearRange_t = [t for t in range(2000,2020+1)]
    paraPath = 'H:/My Drive/PSO_cornBelt'
    paraMode = 'inteval' #'default'#,'defaultV2','eachYear'#'previousYear'#'inteval'#'global'#'eachYear'#'intevalAcc' 
    case = paraMode
    
    # outFolder = 'paraCaliResults/%s/%s'%(algorithm,mode)
    outFolder = 'F:/MidWest_counties/STAT/outVars_case_%s_cornbelt%s'%(case,version)
    if saveRes:
        util.mkdir(outFolder)
    
    # pick valid FIPS
    FIPSList = []
    for f in FIPSList_all:
        if os.path.exists('%s/GPP_%s_corn.pkl'%(GPPpath,f)) & os.path.exists('%s/GPP_%s_soybean.pkl'%(GPPpath,f)):
            FIPSList.append(f)
    validFIPS_GPP,nanYear_dic = util.validGPPcounty()
    FIPSList = list(set(FIPSList).intersection(set(validFIPS_GPP)))
    FIPSList.sort()
    summaryAll = {}
    yield_dic_all = {}
    varMax_dic_all = {}
    varMeanGS_dic_all = {}
    
    for crop in ['corn','soy']:
        print('Processing %s, case: %s...'%(crop,case))
        # obs NASS yield
        NASS_Path = 'F:/MidWest_counties/Yield'
        NASS_yield = util.yieldValidation(NASS_Path)
        if crop == 'corn':
            df_yield = NASS_yield.yield_NASS_corn
            cropType=0
            coef = NASS_yield.coef_C2BUacre(0)
        else:
            df_yield = NASS_yield.yield_NASS_soybean
            cropType=1
            coef = NASS_yield.coef_C2BUacre(1)
        df_yield.drop(['Unnamed: 0'],inplace=True,axis=1)
        loc = [i for i,t in enumerate(df_yield.Year.tolist()) if t in yearRange_t]
        df_yield_ = df_yield.iloc[loc]
        # divide batch
        FIPSbatchList = []
        for n,FIPS in enumerate(FIPSList):
            if n == 0:
                tmp = [FIPS]
            elif n%batchFIPS == 0:
                FIPSbatchList.append(tmp)
                tmp = [FIPS]
            else:
                tmp.append(FIPS)
        FIPSbatchList.append(tmp)
        
        yield_dic = {}
        yield_dic['Year'] = yearRange_t
        varMax_dic = {}
        varMax_dic['Year'] = yearRange_t
        varMeanGS_dic = {}
        varMeanGS_dic['Year'] = yearRange_t
        count = 0
        
        # load para 
        para = util.loadPara(paraPath,crop,algorithm,mode=paraMode)
        paraLoc = para.paraLoc
        
        for batch,FIPSbatch in enumerate(FIPSbatchList):
            
            if crop=='corn':
                cropTypes = [0 for _ in range(len(FIPSbatch))]
            else:
                cropTypes = [1 for _ in range(len(FIPSbatch))]
                
            inputMerged_site = []

            startTime = time.time()
            for FIPS in FIPSbatch:                          
                # load site data
                tmp = []
         
                # load input and Ecosys output data
                inputDataPath = '%s/%s_inputMerged.pkl'%(dataRoot,FIPS)
                inputData = util.load_object(inputDataPath)
                inputMerged_site.append(inputData)
        
            # gen input            
            genEn = makeInput(inputMerged=inputMerged_site,paraMode=paraMode)
            inputEpisode_reset = []
            yieldTimeSeires = {}
            varMaxTimeSeires = {}
            varMeanGSTimeSeires = {}
            
            for year in yearRange_t:
                # load para 
                para_cali = para.getPara(year)
                if para_cali is None:
                    cropParaCali=None
                else:
                    para_list = para_cali[FIPSbatch]
                    cropParaCali=[np.array(para_list).T,paraLoc]
                            
                # make ensemble input
                inputEpisodes = genEn.get(year)
                inputEpisode_reset = genEn.resetDefaultPara(inputEpisodes,cropTypes=cropTypes,cropParaCali=cropParaCali)
                
                inputEpisode_vali, validFIPS = genEn.validCheck(inputEpisode_reset,FIPSbatch)
                inputEpisode_en = np.stack(inputEpisode_vali)
                
                # run
                out = model.run(inputEpisode_en)
                yieldList = out[:,-2,-1]/genEn.y_NormCoef[-1]
                varMax = np.max(np.abs(out),axis=1)/np.array(genEn.y_NormCoef)
                varMeanGS = np.mean(out[:,151:243,:],axis=1)/np.array(genEn.y_NormCoef)
                
                # classify the FIPS         
                if year == yearRange_t[0]:
                    n = 0
                    for s in FIPSbatch:
                        if s in validFIPS:
                            yieldTimeSeires[s] = [yieldList[n]]
                            varMaxTimeSeires[s] = [varMax[n,:]]
                            varMeanGSTimeSeires[s] = [varMeanGS[n,:]]
                            n+=1
                        else:
                            yieldTimeSeires[s] = [None]
                            varMaxTimeSeires[s] = [None]
                            varMeanGSTimeSeires[s] = [None]
                           
                else:
                    n = 0
                    for s in FIPSbatch:
                        if s in validFIPS:
                            yieldTimeSeires[s].append(yieldList[n])
                            varMaxTimeSeires[s].append(varMax[n,:])
                            varMeanGSTimeSeires[s].append(varMeanGS[n,:])
                          
                            n+=1
                        else:
                            yieldTimeSeires[s].append(None)
                            varMaxTimeSeires[s].append(None)
                            varMeanGSTimeSeires[s].append(None)
                            
            for n,t in enumerate(FIPSbatch):            
                yield_dic[t] = yieldTimeSeires[t]
                varMax_dic[t] = varMaxTimeSeires[t]
                varMeanGS_dic[t] = varMeanGSTimeSeires[t]
                
            count+= len(FIPSbatch)
            finishTime = time.time()
            print('%d/%d counties finished, take %.2f s.'%(count,len(FIPSList),finishTime-startTime))
        
        # plot
        yield_df = pd.DataFrame(yield_dic)                   
        
        summary = NASS_vs_ecosys(df_NASS=df_yield_,df_p=yield_df,crop=crop,coef=coef,
                       mode=case)
        
        summaryAll[crop] = summary
        yield_dic_all[crop] = yield_dic
        varMax_dic_all[crop] = varMax_dic
        varMeanGS_dic_all[crop] = varMeanGS_dic
     
    if saveRes:
        util.save_object(varMax_dic_all, '%s/trend_var_max.pkl'%outFolder)
        util.save_object(varMeanGS_dic_all, '%s/trend_var_MeanGS.pkl'%outFolder)
        
    #stat
    for crop in ['corn','soy']:
        
        varList = genEn.y_selectFeatures
        yearAxis = yearRange_t
        
        # max
        target_dic = copy.deepcopy(varMax_dic_all[crop])
        _=target_dic.pop('Year')
        item = 'var_max'
        fig = subplotTrend(target_dic,yearAxis,varList,crop = crop,item=item)
        if saveRes:
            fig.savefig('%s/trend_%s_%s.png'%(outFolder,item,crop))
            
        # varMeanGS
        target_dic = copy.deepcopy(varMeanGS_dic_all[crop])
        _=target_dic.pop('Year')
        item = 'var_mean_growingSeason'
        fig = subplotTrend(target_dic,yearAxis,varList,crop = crop,item=item)
        if saveRes:
            fig.savefig('%s/trend_%s_%s.png'%(outFolder,item,crop))