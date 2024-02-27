# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 10:13:44 2024

@author: yang8460
"""

import numpy as np
import time
import KGDA_Networks as net
import torch
import KGDA_util as util
import os, glob
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib
from KFs.ensemble_kalman_filter_torch import EnsembleKalmanFilter_parallel_v4_UpdatePara as EnKF
import scipy.signal
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['figure.dpi'] = 300

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device}" " is available.")

class RSobs():
    def __init__(self,data_dir):
        self.data_dir = data_dir
        self.loadGPPdata()
        self.statistic()
        
    def loadGPPdata(self):
        hList = [10,11]
        vList = [4,5]
        dataList = []
        n=0
        pklFile = '%s/GPPextracted_25296.pkl'%self.data_dir
        if os.path.exists(pklFile):
            self.dataAll = util.load_object(pklFile)
        else:
            for h in hList:
                for v in vList:
                    outFile = '%s/GPPextracted_25296_tile_h%2dv%02d.csv'%(self.data_dir,h,v)
                    tmp = pd.read_csv(outFile)
                    tmp.drop(['Unnamed: 0'],axis=1,inplace=True)
                    if n>0:
                        tmp.drop(['Date'],axis=1,inplace=True)
                    n+=1
                    dataList.append(tmp)
            self.dataAll = pd.concat(dataList,axis=1)
            self.dataAll['Date'] = pd.to_datetime(self.dataAll['Date'],format='%Y-%m-%d')
            util.save_object(self.dataAll, pklFile)
        print('RS data loaded!')
    
    def getObs(self, site):
        obsDate = self.dataAll['Date'].tolist()
        obs = self.dataAll[site].tolist()
        return obsDate, obs

    def statistic(self):
        dateList = self.dataAll.Date.tolist()
        startY = dateList[0].strftime("%Y")
        endY = dateList[-1].strftime("%Y")
        self.yearRange = [y for y in range(int(startY),int(endY)+1)]
        
        self.dataAll_year = []
        indexS = []
        indexE = []
        for i,d in enumerate(dateList):
            if i==0:
                indexS.append(i)
                currentY = dateList[indexS[-1]].strftime("%Y")
            if d.strftime("%Y") != currentY:
                indexE.append(i-1)
                self.dataAll_year.append(self.dataAll.iloc[indexS[-1]:indexE[-1]+1])
                
                indexS.append(i)
                currentY = dateList[indexS[-1]].strftime("%Y")
            if d == dateList[-1]:
                indexE.append(i)
                self.dataAll_year.append(self.dataAll.iloc[indexS[-1]:indexE[-1]+1])
            
    def plotGPP(self,CDL,sites):
        self.GPPmean_df = pd.DataFrame()
        self.GPPmean_df['Year'] = self.yearRange

        GPP_sum_mean_corn = []
        GPP_sum_mean_soybean = []
        GPP_max_mean_corn = []
        GPP_max_mean_soybean = []
        
        for year,yearly in zip(self.yearRange,self.dataAll_year): 
  
            CornSite = CDL.getCornSite(year).tolist()
            soybeanSite = CDL.getSoybeanSite(year).tolist()
            GPP_corn_year = np.array(yearly[CornSite])
            GPP_soybean_year = np.array(yearly[soybeanSite])
            # for s in sites:
            #     cropType=CDL.get(year,s)
            #     if cropType == 0:
            #         GPPmean_corn_year.append(np.sum(yearly[s]))
            #     elif cropType == 1:
            #         GPPmean_soybean_year.append(np.sum(yearly[s]))
            #     else:
            #         pass
            GPP_sum_mean_corn.append(np.mean(np.sum(GPP_corn_year,axis=0)))
            GPP_sum_mean_soybean.append(np.mean(np.sum(GPP_soybean_year,axis=0)))
            GPP_max_mean_corn.append(np.mean(np.max(GPP_corn_year,axis=0)))
            GPP_max_mean_soybean.append(np.mean(np.max(GPP_soybean_year,axis=0)))
            
        self.GPPmean_df['GPPacc_corn'] = GPP_sum_mean_corn
        self.GPPmean_df['GPPacc_soybean'] = GPP_sum_mean_soybean
        self.GPPmean_df['GPPmax_corn'] = GPP_max_mean_corn
        self.GPPmean_df['GPPmax_soybean'] = GPP_max_mean_soybean
        
        plt.figure(figsize=(10,5))
        plt.plot(self.GPPmean_df['Year'],self.GPPmean_df['GPPacc_corn'],'r-',label='SLOPE GPP accumulated, corn ')
        plt.plot(self.GPPmean_df['Year'],self.GPPmean_df['GPPacc_soybean'],'g-',label='SLOPE GPP accumulated, soybean')
        plt.legend()
        
        plt.figure(figsize=(10,5))
        plt.plot(self.GPPmean_df['Year'],self.GPPmean_df['GPPmax_corn'],'r-',label='SLOPE GPP max, corn')
        plt.plot(self.GPPmean_df['Year'],self.GPPmean_df['GPPmax_soybean'],'g-',label='SLOPE GPP max, soybean')
        plt.legend()
        
class makeInput_ensemble_parallel():  
    def __init__(self, inputMerged,startYear=2000):
        
        self.inputMerged=inputMerged
        self.startYear = startYear
        
        ## in & out
        self.cropParaList = ['VCMX','CHL4','GROUPX','STMX','GFILL','SLA1']
        self.cropPara_index = [7,8,9,10,11,12]
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
    
    def resetDefaultPara(self,inputEpisodes,cropTypes,cropParaDefaults=None):
        inputEpisode_reset_list = []
        n=0
        for inputEpisode,cropType in zip(inputEpisodes,cropTypes):
            if inputEpisode is None:
                inputEpisode_reset_list.append(None)
            else:
                inputEpisode_reset = inputEpisode.copy()
                inputEpisode_reset['CropType'] = cropType
                if cropParaDefaults is None:
                    for v, vi in zip(self.cropParaDefaults[cropType],self.cropParaList):
                        inputEpisode_reset[vi] = v
                else:
                    for v, vi in zip(cropParaDefaults[n,:],self.cropParaList):
                        inputEpisode_reset[vi] = v  
                if cropType==0:
                    inputEpisode_reset['Fertilizer'] = self.fert
                    dateP = datetime.date(self.year,self.plantingDate_corn[0],self.plantingDate_corn[1])
                else:
                    inputEpisode_reset['Fertilizer'] = 0
                    dateP = datetime.date(self.year,self.plantingDate_soybean[0],self.plantingDate_soybean[1])
                          
                # # set growing season          
                # delta = datetime.timedelta(np.random.randint(-self.DisturbDays,self.DisturbDays))
                # dateP_disturb = dateP + delta
                # dateP_disturb_DOY = dateP_disturb.timetuple().tm_yday        
                dateP_DOY = dateP.timetuple().tm_yday
                tmp = np.array(inputEpisode_reset['GrowingSeason'].copy())
                tmp[dateP_DOY-1:] = 1
                inputEpisode_reset['GrowingSeason'] = tmp
                inputEpisode_reset_list.append(inputEpisode_reset)
            n+=1
        return inputEpisode_reset_list
        
    def disturbPara(self,inputEpisodes, ensemble_n = 100,FIPSbatch=None,disturbPD=False,cropTypes=None):
        np.random.seed(0)
        daily_En_list = []
        validFIPS = []
        for inputEpisode,FIPS,c in zip(inputEpisodes,FIPSbatch,cropTypes):
            if inputEpisode is None:
                continue
            validFIPS.append(FIPS)
            inputEpisode=np.array(inputEpisode).astype(np.float32)
            
            # disturb parameters
            u_para_default = inputEpisode[0,7:]
            CV = [0.1]*len(u_para_default)
            if c == 0:
                CV[0] = 0 # corn, don't disturb VCMX
            else:
                CV[1] = 0 # soybean, don't disturb CHL4
            # CV = [0]*17 # Coefficient of Variation
            std2_0 = list((np.asarray(u_para_default) * np.asarray(CV))**2)
            P_u_para = np.diag(std2_0)
            ensemble_paras = list(np.random.multivariate_normal(mean=u_para_default, cov=P_u_para, size=ensemble_n))
            
            # disturb planting date
            pD = list(inputEpisode[:,5]).index(1)+1
            ensemble_PD = list(np.random.normal(loc=pD,scale=(0.05*pD), size=ensemble_n))
            daily_in_ensemble = []
            
            # reconstruct the input
            for i in range(ensemble_n):
                tmp = inputEpisode.copy()
                for n,p in enumerate(ensemble_paras[i]):
                    tmp[:,n+7] = p
                    
                # growing season
                season = np.zeros((tmp.shape[0])).astype(np.float32)
                season[int(ensemble_PD[i])-1:] = 1
                tmp[:,5] = season
                
                daily_in_ensemble.append(tmp)
            
            daily_in_ensemble_array = np.array(daily_in_ensemble)
            daily_En_list.append(daily_in_ensemble_array)
        return daily_En_list, validFIPS

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class ecoNet_env():
    def __init__(self,model):
        self.model = model
        self.reset()
        
    def reset(self):
        self.hidden_state = None
        self.previousLAI = None
        
    def steprun(self, x, updateState=None):
        '''
        
        Parameters
        ----------
        x : shape:[sites,ensemble,dt,features]

        Returns
        -------
        prediction

        '''   
        LAIloc=7
        
        # compact batch
        xList = list(x)
        self.siteN = len(xList)
        self.enN = x.shape[1]
        xCompact = np.concatenate(xList,axis=0)
        if not updateState is None:
            updateState_com = np.concatenate(list(updateState),axis=0)
        else:
            updateState_com = None
        # run
        yhat,self.hidden_state = self.model(torch.tensor(xCompact.astype(np.float32)).to(device),
                                            self.hidden_state,updateState=updateState_com,initLAI=self.previousLAI)
        self.previousLAI = yhat[:,0,LAIloc].view([-1,1])

        # decompact batch
        yhat_decompact = torch.split(yhat,self.enN,dim=0)
        return yhat_decompact

class enRun():
    def __init__(self, obsType, MaskedIndex, stateN):
        self.obsType=obsType
        self.MaskedIndex=MaskedIndex
        self.stateN=stateN
        self.H = torch.unsqueeze(torch.zeros(self.stateN, dtype=torch.float32),dim=1).to(device)
        # self.H[obsType] = 1
        self.H[0] = 1
        
    def oneRun(self, inputFeature,upInfo, ecoNet, openLoop=False, measurements =None, R=None,cellRange=None,MaskCells=None):
        if openLoop:
            
            ecoNet.model.eval()
            yhat_list = []
            for i in range(inputFeature.shape[2]):
                dailyIn = inputFeature[:,:,i,:][:,:,np.newaxis,:].astype(np.float32)   
                
                ## predict
                yhat = torch.stack(ecoNet.steprun(dailyIn))
                yhat_list.append(yhat.detach().cpu().numpy())
            
            # decompact batch
            predictions_decompact = np.squeeze(np.stack(yhat_list)).transpose(1,2,0,3)
            return None,None,None,None,predictions_decompact
        else:
    
            ecoNet.model.eval()
            
            # assimilation
            self.siteN = inputFeature.shape[0]
            ensemble_n = inputFeature.shape[1]
            P0 = np.diag([0.])
            DA_enkf = EnKF(x=np.zeros(len(upInfo['updateParaList'])+1), P=P0, 
                           dim_z=1, N=ensemble_n, H = self.H, fx = ecoNet.steprun,cellRange=cellRange)
            DA_enkf.R = R.copy()
            
            xs_enkf = []
            sigmas_enkf = []
            P_enkf = []
            K_enkf = []
            out_model = []
            ass_n = 0
            if measurements is None:
                z_DAP = []
            else:
                z_DAP = measurements[0]
                zs = measurements[1]
            DAP = 1
            updateHidden = False
            for i in range(inputFeature.shape[2]):
                dailyIn = inputFeature[:,:,i,:][:,:,np.newaxis,:].astype(np.float32)   
                if i==0:
                    DA_enkf.setPara(dailyIn, paraIndex = upInfo['updatePara_index'])
                    
                ## predict
                DA_enkf.predict(dailyIn,updateHidden,MaskCells =MaskCells, stateIndex = [self.obsType], paraIndex = upInfo['updatePara_index'])
                updateHidden = False
                
                ## update
                if DAP in z_DAP:
                    # print('debug,update phase, DAP is %s'%DAP)
                    DA_enkf.update(zList=[np.array([z[ass_n]]) for z in zs],MaskedIndex=self.MaskedIndex)
                    updateHidden = True
                    ass_n += 1
                    K_enkf.append(DA_enkf.K_batch.cpu().detach().numpy().copy())
                  
                ## record
                xs_enkf.append(DA_enkf.x_batch.cpu().detach().numpy().copy())
                P_enkf.append(DA_enkf.P_batch.cpu().detach().numpy().copy())   
                sigmas_enkf.append(DA_enkf.sigmas_batch.cpu().detach().numpy().copy())
                out_model.append(torch.stack(DA_enkf.out).cpu().detach().numpy().copy())
                DAP += 1
            if self.siteN == 1:
                sigmas_enkf = np.squeeze(np.array(sigmas_enkf)).transpose(1,0,2)[np.newaxis,:,:,:]
                out_model = np.squeeze(np.array(out_model)).transpose(1,0,2)[np.newaxis,:,:,:]
            else:
                sigmas_enkf = np.squeeze(np.array(sigmas_enkf)).transpose(1,2,0,3)
                out_model = np.concatenate(out_model,axis=2)
            return xs_enkf,P_enkf,K_enkf,sigmas_enkf,out_model

def inperiod(d,periodDate):
    if (d>=periodDate[0]) & (d<=periodDate[1]):
        return True
    else:
        return False
    
def plot_test_series(sList,labelList,fmtList,outFolder=None,saveFig=False,note='',ylabel='',title='',dateCol = None, scale = 1.0, periodDate=None):
    color_list = ['darkgreen','g','darkred','r','m','b','k','y','c','sienna','navy','grey']
    sList=[np.array(t).astype(float) for t in sList]
       
    fig = plt.figure(figsize=(7,5))
    for i,s in enumerate(sList):
        s[s==-9999]=np.nan
        s[s<-999]=np.nan
        if dateCol is None:
            plt.plot(s*scale, color=color_list[i],  label=labelList[i])
        else:
            if periodDate==None:
                x=dateCol[i]
            else:
                x = np.array([t for t in dateCol[i] if inperiod(t, periodDate)])
                if len(s.shape)==1:
                    s = np.array([t2 for t,t2 in zip(dateCol[i],s) if inperiod(t, periodDate)])
                else:
                    s = np.array([s[:,i] for i,t in enumerate(dateCol[i]) if inperiod(t, periodDate)])
                    s = s.transpose(1,0)
            if fmtList[i] =='line':    
                plt.plot_date(x,s*scale, color=color_list[i],  label=labelList[i], linestyle='-' ,fmt='None',alpha=1)
            elif fmtList[i] =='marker':   
                plt.plot_date(x,s*scale, color=color_list[i],  label=labelList[i], fmt='*', alpha=.6)
            elif fmtList[i] =='marker_circle':   
                plt.plot_date(x,s*scale, color=color_list[i],  label=labelList[i], fmt='o', alpha=1,markersize=10,fillstyle='none')
            elif fmtList[i] =='ensemble':
                for t in range(s.shape[0]):
                    if t==0:
                        plt.plot_date(x,s[t,:],color = color_list[i] , linestyle='-', fmt='None', alpha=.05,linewidth=0.5)#, label=labelList[i])
                    else:
                        plt.plot_date(x,s[t,:],color = color_list[i] , linestyle='-', fmt='None', alpha=.05,linewidth=0.5)
    plt.legend(fontsize = 12,frameon=False,loc="upper left")
    plt.xlabel('Date',fontsize = 16)
    if ylabel == 'GPP_daily':
        ylabel = 'GPP gC/m2'
    elif ylabel == 'Reco_daily':
        ylabel = 'Reco gC/m2'
        
    plt.ylabel(ylabel,fontsize = 16)    
    plt.title(title)
    return fig
    
def ensembleProcess_parallel(predictions,i,inputData=None):
    y_en_list = []
    y_mean_list = []
    y_std_list = []
    for t in list(predictions):
        if inputData is not None:
            tmp = t[:,:,i]/inputData.y_NormCoef[i]
        else:
            tmp = t[:,:,i]
        y_en = (tmp)
        y_mean = np.mean(tmp,axis=0)
        y_std = np.std(tmp,axis=0)
        y_en_list.append(y_en)
        y_mean_list.append(y_mean)
        y_std_list.append(y_std)
    # y_en = np.concatenate(y_en,axis=1)
    # y_mean = np.concatenate(y_mean,axis=0) 
    # y_std = np.concatenate(y_std,axis=0)
    return y_en_list, y_mean_list, y_std_list 

def ensembleProcess(predictions,i,inputData,para=False):
    t=predictions
    if para:
        tmp = t[:,:,i]
    else:
        tmp = t[:,:,i]/inputData.y_NormCoef[i]
    y_en = (tmp)
    y_mean = np.mean(tmp,axis=0)
    y_std = np.std(tmp,axis=0)

    return y_en, y_mean, y_std  

def visualizeResult(i,predictions_all,inputData,periodDate,mode='noObs',obs_RS=None,obs_date_RS=None, yearS = 2000,title='',ylabel=None, para=False):
           
    y_en, y_mean, y_std = ensembleProcess(predictions_all,i,inputData,para)
    dateList = [datetime.datetime(yearS,1,1)+datetime.timedelta(days=t) for t in range(len(y_mean))]
    
    if mode == 'noObs':      
        fig=plot_test_series(sList=[y_mean,y_en],labelList=['Data Assimilation Mean','Data Assimilation Ensemble'],fmtList = ['line','ensemble'],
                                ylabel=ylabel,title=title,dateCol =[dateList,dateList],periodDate=periodDate)

    elif mode == 'RSobs':
        fig=plot_test_series(sList=[y_mean,y_en, obs_RS],
                         labelList=['Data Assimilation Mean','Data Assimilation Ensemble','SLOPE GPP'],fmtList = ['line','ensemble','marker_circle'],
                                ylabel=ylabel,title=title,dateCol =[dateList,dateList,obs_date_RS],periodDate=periodDate)
    return fig

def visualizeResult_m(i,predictions_all,predictions_op,inputData,periodDate,mode='noObs',obs_RS=None,obs_date_RS=None,ylabel=None, yearS = 2000,title=''):

    y_en, y_mean, y_std = ensembleProcess(predictions_all,i,inputData)
    dateList = [datetime.datetime(yearS,1,1)+datetime.timedelta(days=t) for t in range(len(y_mean))]

    y_en_op, y_mean_op, y_std_op = ensembleProcess(predictions_op,i,inputData) 
    if mode == 'noObs':      
        fig=plot_test_series(sList=[y_mean_op,y_en_op,y_mean,y_en],labelList=['Without DA','Open-loop Ensemble','Data Assimilation (DA)','Data Assimilation Ensemble'],
                             fmtList = ['line','ensemble','line','ensemble'],
                               ylabel=ylabel,title=title,dateCol =[dateList,dateList,dateList,dateList],periodDate=periodDate)

    elif mode == 'RSobs':
        fig=plot_test_series(sList=[y_mean_op,y_en_op,y_mean,y_en, obs_RS],
                         labelList=['Without DA','Open-loop Ensemble','Data Assimilation (DA)','Data Assimilation Ensemble','SLOPE GPP'],
                         fmtList = ['line','ensemble','line','ensemble','marker_circle'],
                                ylabel=ylabel,title=title,dateCol =[dateList,dateList,dateList,dateList,obs_date_RS],periodDate=periodDate)
    elif mode == 'LAIobs':
        fig=plot_test_series(sList=[y_mean_op,y_en_op,y_mean,y_en, obs_RS],
                         labelList=['Without DA','Open-loop Ensemble','Data Assimilation (DA)','Data Assimilation Ensemble','Ground Truth LAI'],
                         fmtList = ['line','ensemble','line','ensemble','marker_circle'],
                                ylabel=ylabel,title=title,dateCol =[dateList,dateList,dateList,dateList,obs_date_RS],periodDate=periodDate)
        
    return fig

def appendYield(yieldTimeSeires,yieldYear,index):
    if len(yieldYear) == 0:
        yieldTimeSeires[index].append(np.nan)
    else:
        yieldTimeSeires[index].append(np.mean(yieldYear))
    return yieldTimeSeires

def obsExtrat(obs_smooth_dic,FIPS,measDOY,year,coef,startYear=2000):
    obs = obs_smooth_dic[FIPS][year-startYear]
    return [obs[t-1]*coef for t in measDOY]

def plotParaTimeSeries(para_3I_dic_path,para='GROUPX',FIPS='17001',crop='corn'):
    para_3I_dic = util.load_object(para_3I_dic_path)
    y = para_3I_dic[para][FIPS]
    fig = plt.figure(figsize=(12,5))
    plt.plot(y)
    plt.title('%s: %s'%(crop,para))
    return fig

def maskinfo(maskMode):
    if maskMode == 'openLoop':
        openLoop=True#False
        MaskCells = None
        MaskedIndex = None
    else:
        openLoop=False
        if maskMode == 'maskNone':
            MaskCells = None
            MaskedIndex = None
        elif maskMode == 'maskc3c4':
            MaskCells = [2,3]
            MaskedIndex = [4,5,6,7,8] # don't update this state variable, or None
        elif maskMode == 'maskc1c3c4':
            MaskCells = [0,2,3]
            MaskedIndex = [0,4,5,6,7,8] # don't update this state variable, or None
        elif maskMode == 'maskc1c2c3':
            MaskCells = [0,1,2]
            MaskedIndex = [0,1,2,3,4,5,6] # don't update this state variable, or None
        elif maskMode == 'maskc2c3':
            MaskCells = [1,2]
            MaskedIndex = [1,2,3,4,5,6] # don't update this state variable, or None
        elif maskMode == 'maskc3':
            MaskCells = [2]
            MaskedIndex = [4,5,6] # don't update this state variable, or None
        elif maskMode == 'maskc1c2c3yield':
            MaskCells = [0,1,2]
            MaskedIndex = [0,1,2,3,4,5,6,8] # don't update this state variable, or None
        elif maskMode == 'maskc2c3yield':
            MaskCells = [1,2]
            MaskedIndex = [1,2,3,4,5,6,8] # don't update this state variable, or None
        elif maskMode == 'maskc4':
            MaskCells = [3]
            MaskedIndex = [7,8] # don't update this state variable, or None
        elif maskMode == 'maskc3yield':
            MaskCells = [2]
            MaskedIndex = [4,5,6,8] # don't update this state variable, or None
        elif maskMode == 'maskc1':
            MaskCells = [0]
            MaskedIndex = [0] # don't update this state variable, or None
        elif maskMode == 'maskc1yield':
            MaskCells = [0]
            MaskedIndex = [0,8] # don't update this state variable, or None
        elif maskMode == 'maskLAI':
            MaskCells = None
            MaskedIndex = [7]    
        elif maskMode == 'maskYield':
            MaskCells = None
            MaskedIndex = [8] # don't update this state variable, or None
            
    return MaskCells,MaskedIndex,openLoop

if __name__ == '__main__':
    
    ## setting
    FIPSList = ['17001','17003']
    yearRange = [2001,2005]    
    case='case1'  # obs type
    # para to update
    upInfo = {}
    upInfo['updateParaList'] = ['VCMX','CHL4','GROUPX','STMX','GFILL','SLA1']#,'Fertilizer']
    upInfo['updatePara_index'] = [7,8,9,10,11,12]#,-1]
    saveFig = True
    saveRes = True
    disturbPD = True
    # inherit the updated para for the next year or not
    heritPara = True
    ensemble_n = 100
    obsIntertval = 10
    if heritPara:
        window = 5  # moving window to ave parameters     
    obsType = 2 # GPP   
    dataRoot = r'demoData/Midwest_counties'
    
    # process the setting
    periodDate=[datetime.datetime.strptime('%s0101'%(yearRange[0]),'%Y%m%d'),
                                              datetime.datetime.strptime('%s1231'%(yearRange[1]),'%Y%m%d')]
    yearSpan = np.arange(yearRange[0],yearRange[1]+1)
    GPPpath = dataRoot
    countyPathes = glob.glob('%s/*.pkl'%dataRoot)
    FIPSList_all = [t.split('\\')[-1].split('_')[0] for t in countyPathes]
    now = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    
    # pick valid FIPS
    FIPSList = []
    for f in FIPSList_all:
        if os.path.exists('%s/GPP_%s_corn.pkl'%(GPPpath,f)) & os.path.exists('%s/GPP_%s_soybean.pkl'%(GPPpath,f)):
            FIPSList.append(f)
     
    # load networks
    maskMode = 'maskYield'
    mode='paraPheno_c2'
    batchFIPS = 2
    input_dim = 21
    output_dim=[1,3,3,2]
    hidden_dim = 64
    cellRange = [[0,output_dim[0]],
                 [output_dim[0],np.sum(output_dim[0:2])],
                 [np.sum(output_dim[0:2]),np.sum(output_dim[0:3])],
                 [np.sum(output_dim[0:3]),np.sum(output_dim[0:4])]]
    
  
    note = 'obsInterval%s_%s'%(obsIntertval,case)
    projectName = '%s-%s'%(note,now)
    outFolder = 'Result/updatePara/%s'%(projectName)

    # get the mask information. MaskCells means do not update the hidden state of that cell, MaskedIndex means to set the Kalman gain of that state var to zero
    MaskCells,MaskedIndex,openLoop = maskinfo(maskMode)
    model = to_device(net.KG_ecosys(input_dim=input_dim,hidden_dim=hidden_dim,
                                                output_dim=output_dim,mode=mode), device)

    # load network
    modelName = 'demo_model_cornBelt20299_state_dict.pth'
    model.load_state_dict(torch.load(modelName))     
    ecoNet = ecoNet_env(model)
    
    # initialize the class
    run = enRun(obsType=obsType,MaskedIndex=MaskedIndex,stateN=len(upInfo['updateParaList'])+1)
    R = np.diag([0.01**2])  # measurement cov matrix
    
    # run model for corn and soybean
    for crop in ['corn','soy']:

        yield_3I_dic = {}
        yield_3I_dic['Year'] = yearSpan
        
        # episodes
        eps = 0
        
        # measurements DOY
        measDOY = [i for i in range(160, 220, obsIntertval)]  # Jun.1 = Sep.1
        measDOY = [i for i in range(130, 300, obsIntertval)] 
        
        # county ID for each batch
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
        
        # loop for each batch
        count = 0
        GPPmax_3I_dic = {}
        GPPsum_3I_dic = {}
        para_3I_dic = {}
        for t in upInfo['updateParaList']:
            para_3I_dic[t] = {}
        for batch,FIPSbatch in enumerate(FIPSbatchList):
            inputMerged_site = []
            obs_smooth_dic = {}
            startTime = time.time()
            for FIPS in FIPSbatch:     
                     
                # load input data for a county
                inputDataPath = '%s/%s_inputMerged.pkl'%(dataRoot,FIPS)
                inputData = util.load_object(inputDataPath)
                inputMerged_site.append(inputData)
        
                # fetch and process obs data
                if crop=='corn':
                    tmp = util.load_object('%s/GPP_%s_%s.pkl'%(GPPpath,FIPS,crop))
                else:
                    tmp = util.load_object('%s/GPP_%s_%sbean.pkl'%(GPPpath,FIPS,crop))
                obs = [tmp[t] for t in range(yearRange[0],yearRange[1]+1)]

                obs_coef = 0.6
                obs_smooth_dic[FIPS] = [scipy.signal.savgol_filter(t,21,3)*obs_coef if t is not None else None for t in obs]
    
            # make the ensemble inputs
            genEn = makeInput_ensemble_parallel(inputMerged=inputMerged_site)
            if crop=='corn':
                cropTypes = [0 for _ in range(len(FIPSbatch))]
            else:
                cropTypes = [1 for _ in range(len(FIPSbatch))]
            
            # run for each year
            measList = []
            measDateList = []
            yieldTimeSeires = {}
            GPPmaxTimeSeires = {}
            GPPsumTimeSeires = {}
            ParaTimeSeires = {}
            for t in upInfo['updateParaList']:
                ParaTimeSeires[t]={}
            
            ParaMeanArray = np.tile(genEn.cropParaDefaults[cropTypes[0]],(len(FIPSbatch),1))  # placeholder
            ParaMeanArrayList = [ParaMeanArray.copy()]
            for year in range(yearRange[0],yearRange[1]+1):
                
                # inherit the updated parameters from previous year
                if heritPara:
                    # moving window to ave updated parameters
                    tmp = np.stack(ParaMeanArrayList)
                    if window>1:
                        if tmp.shape[0]<window:
                            ParaMeanArray_mw = np.mean(tmp,axis=0)
                        else:
                            ParaMeanArray_mw = np.mean(tmp[-window:,:,:],axis=0)
                    else:
                        ParaMeanArray_mw = ParaMeanArray  
                
                # fetch data for a specific year
                inputEpisodes = genEn.get(year)
                
                # intialize the parameter
                if heritPara:
                    # use the updated para
                    if year > yearRange[0]:    
                        inputEpisode_reset = genEn.resetDefaultPara(inputEpisodes,cropTypes=cropTypes,cropParaDefaults=ParaMeanArray_mw)
                    else:
                        inputEpisode_reset = genEn.resetDefaultPara(inputEpisodes,cropTypes=cropTypes)
                else:
                    # use default para
                    inputEpisode_reset = genEn.resetDefaultPara(inputEpisodes,cropTypes=cropTypes)
                
                # perturb parameters
                enList, validFIPS = genEn.disturbPara(inputEpisode_reset,ensemble_n = ensemble_n,FIPSbatch=FIPSbatch,
                                                      disturbPD=disturbPD,cropTypes=cropTypes)
                inputEpisode_en = np.stack(enList)
                ecoNet.reset()
                measDate = [datetime.datetime(year,1,1)+datetime.timedelta(i-1) for i in measDOY]
                measValue = []
                for FIPS in validFIPS:   
                    measValue.append(obsExtrat(obs_smooth_dic,FIPS,measDOY,year,
                                               coef=genEn.y_NormCoef[obsType],startYear=yearRange[0]))

                # run
                measurements = [measDOY, measValue]

                xs_enkf,P_enkf,K_enkf,sigmas_enkf,out_model = run.oneRun(inputFeature=inputEpisode_en, 
                                                            upInfo=upInfo, ecoNet=ecoNet, 
                                                            openLoop=openLoop,measurements=measurements,R=R,cellRange=cellRange,MaskCells=MaskCells)

                ecoNet.reset()
                _,_,_,_,sigmas_enkf_op = run.oneRun(inputFeature=inputEpisode_en, 
                                                            upInfo=upInfo, ecoNet=ecoNet, 
                                                            openLoop=True,measurements=measurements,R=R,
                                                            cellRange=cellRange,MaskCells=None)
            
                # log the measument info
                if measurements is not None:
                    measList.extend(measurements[1])
                    measDateList.extend([datetime.datetime.strptime('%d%03d'%(year,t),'%Y%j') for t in measurements[0]])
                
                # yield statistics
                _, y_mean_list, GPP_std_list = ensembleProcess_parallel(out_model,i=-1,inputData=genEn)
                # GPP statictics
                _, GPP_mean_list, y_std_list = ensembleProcess_parallel(out_model,i=2,inputData=genEn)
                
                # log the para info
                ParaMean = {}
                ParaMeanList = []
                loc_validFIPS = [i  for i,t in enumerate(FIPSbatch) if t in validFIPS]
                for i,t in enumerate(upInfo['updateParaList']):
                    tmp = ensembleProcess_parallel(sigmas_enkf,i=i+1)[1]
                    ParaMean[t] = tmp
                    ParaMeanList.append(np.stack(tmp)[:,-1])
                ParaMeanArray_vali = np.stack(ParaMeanList).transpose((1,0))
                tmp = ParaMeanArray[loc_validFIPS,:].copy()
                tmp[:,list(np.array(upInfo['updatePara_index'])-7)] = ParaMeanArray_vali
                ParaMeanArray[loc_validFIPS,:] = tmp
                ParaMeanArrayList.append(ParaMeanArray.copy())  
                
                # plot example fig for the first episode
                if eps<1:
                    i=2
                    obs_RS = np.array(measList[eps]).astype(np.float32)/genEn.y_NormCoef[i]
                    obs_date_RS = measDate
                    fig1 = visualizeResult_m(i=i,predictions_all=out_model[0,:,:,:],predictions_op=sigmas_enkf_op[0,:,:,:],inputData=genEn,periodDate=periodDate,mode='RSobs'
                                    ,obs_RS=obs_RS,obs_date_RS=obs_date_RS,yearS=year,ylabel=genEn.y_selectFeatures[i])
                    i=0
                    fig2 = visualizeResult_m(i=i,predictions_all=out_model[0,:,:,:],predictions_op=sigmas_enkf_op[0,:,:,:],inputData=genEn,periodDate=periodDate,yearS=year
                                           ,ylabel=genEn.y_selectFeatures[i])
                    
                    i=-2
                    fig3 = visualizeResult_m(i=i,predictions_all=out_model[0,:,:,:],predictions_op=sigmas_enkf_op[0,:,:,:],inputData=genEn,periodDate=periodDate,yearS=year
                                           ,ylabel=genEn.y_selectFeatures[i])
                    
                    i=-1
                    fig3 = visualizeResult(i=i,predictions_all=out_model[0,:,:,:],inputData=genEn,periodDate=periodDate,yearS=year
                                           ,ylabel=genEn.y_selectFeatures[i])
                    
                    i=1
                    fig4 = visualizeResult(i=i,predictions_all=sigmas_enkf[0,:,:,:],inputData=genEn,periodDate=periodDate,yearS=year
                                            ,ylabel=upInfo['updateParaList'][i-1],para=True)
                    i=2
                    fig5 = visualizeResult(i=i,predictions_all=sigmas_enkf[0,:,:,:],inputData=genEn,periodDate=periodDate,yearS=year
                                            ,ylabel=upInfo['updateParaList'][i-1],para=True)
                    i=3
                    fig6 = visualizeResult(i=i,predictions_all=sigmas_enkf[0,:,:,:],inputData=genEn,periodDate=periodDate,yearS=year
                                            ,ylabel=upInfo['updateParaList'][i-1],para=True)
                    i=4
                    fig7 = visualizeResult(i=i,predictions_all=sigmas_enkf[0,:,:,:],inputData=genEn,periodDate=periodDate,yearS=year
                                            ,ylabel=upInfo['updateParaList'][i-1],para=True)
                    i=5
                    fig8 = visualizeResult(i=i,predictions_all=sigmas_enkf[0,:,:,:],inputData=genEn,periodDate=periodDate,yearS=year
                                            ,ylabel=upInfo['updateParaList'][i-1],para=True)
                    i=6
                    fig9 = visualizeResult(i=i,predictions_all=sigmas_enkf[0,:,:,:],inputData=genEn,periodDate=periodDate,yearS=year
                                            ,ylabel=upInfo['updateParaList'][i-1],para=True)
                    i=-1
                    fig10 = visualizeResult(i=i,predictions_all=sigmas_enkf[0,:,:,:],inputData=genEn,periodDate=periodDate,yearS=year
                                            ,ylabel=upInfo['updateParaList'][i],para=True)
                    # if saveRes:
                    #     if not os.path.exists(outFolder):
                    #         os.makedirs(outFolder)
                    #     fig1.savefig('%s/test_scatter_GPP_%s_%s.png'%(outFolder,eps,crop))
                    #     fig2.savefig('%s/test_scatter_DVS_%s_%s.png'%(outFolder,eps,crop))
                    #     fig3.savefig('%s/test_scatter_Yield_%s_%s.png'%(outFolder,eps,crop))
                    #     fig4.savefig('%s/test_scatter_%s_%s_%s.png'%(outFolder,upInfo['updateParaList'][1-1],eps,crop))
                    #     fig5.savefig('%s/test_scatter_%s_%s_%s.png'%(outFolder,upInfo['updateParaList'][2-1],eps,crop))
                    #     fig6.savefig('%s/test_scatter_%s_%s_%s.png'%(outFolder,upInfo['updateParaList'][3-1],eps,crop))
                    #     fig7.savefig('%s/test_scatter_%s_%s_%s.png'%(outFolder,upInfo['updateParaList'][4-1],eps,crop))
                    #     fig8.savefig('%s/test_scatter_%s_%s_%s.png'%(outFolder,upInfo['updateParaList'][5-1],eps,crop))
                    #     fig9.savefig('%s/test_scatter_%s_%s_%s.png'%(outFolder,upInfo['updateParaList'][6-1],eps,crop))
                        # fig10.savefig('%s/test_scatter_%s_%s_%s.png'%(outFolder,upInfo['updateParaList'][7-1],eps,crop))
                        
                eps+=1
                
                # classify the FIPS         
                if year == yearRange[0]:
                    n = 0
                    for s in FIPSbatch:
                        if s in validFIPS:
                            yieldTimeSeires[s] = [y_mean_list[n][-1]]
                            tmp = GPP_mean_list[n]
                            GPPmaxTimeSeires[s] = [np.mean(np.sort(tmp)[-20:])] # mean of first 20 of max
                            GPPsumTimeSeires[s] = [np.sum(tmp[151:243])]  # Jun.1- Sep.1
                            for t in upInfo['updateParaList']:
                                ParaTimeSeires[t][s] = list(ParaMean[t][n])
                            n+=1
                        else:
                            yieldTimeSeires[s] = [None]
                            GPPmaxTimeSeires[s] = [None]
                            GPPsumTimeSeires[s] = [None]   
                            for t in upInfo['updateParaList']:
                                ParaTimeSeires[t][s] = [np.nan]*sigmas_enkf.shape[2]
                else:
                    n = 0
                    for s in FIPSbatch:
                        if s in validFIPS:
                            yieldTimeSeires[s].append(y_mean_list[n][-1])
                            tmp = GPP_mean_list[n]
                            GPPmaxTimeSeires[s].append(np.mean(np.sort(tmp)[-20:]))
                            GPPsumTimeSeires[s].append(np.sum(tmp[151:243])) # Jun.1- Sep.1
                            for t in upInfo['updateParaList']:
                                ParaTimeSeires[t][s].extend(ParaMean[t][n])
                            n+=1
                        else:
                            yieldTimeSeires[s].append(None)
                            GPPmaxTimeSeires[s].append(None)
                            GPPsumTimeSeires[s].append(None)
                            for t in upInfo['updateParaList']:
                                ParaTimeSeires[t][s].extend([np.nan]*sigmas_enkf.shape[2])
                                   
            for n,t in enumerate(FIPSbatch):            
                yield_3I_dic[t] = yieldTimeSeires[t]
                GPPmax_3I_dic[t] = GPPmaxTimeSeires[t]
                GPPsum_3I_dic[t] = GPPsumTimeSeires[t]
            for k in para_3I_dic.keys():
                para_3I_dic[k].update(ParaTimeSeires[k])
            finishTime = time.time()
            count+= len(FIPSbatch)
            print('%d/%d counties finished, take %.2f s.'%(count,len(FIPSList),finishTime-startTime))
                
        # save result
        yield_3I = pd.DataFrame(yield_3I_dic)  
        GPPmax_3I = pd.DataFrame(GPPmax_3I_dic)  
        GPPsum_3I =pd.DataFrame(GPPsum_3I_dic)  
        if saveRes:
  
            if not os.path.exists(outFolder):
                os.makedirs(outFolder)
            if crop=='corn':
                yield_3I.to_csv('%s/yield_3I_%s.csv'%(outFolder,crop))

            else:
                yield_3I.to_csv('%s/yield_3I_soybean.csv'%(outFolder))
            util.save_object(para_3I_dic,'%s/para_3I_%s.pkl'%(outFolder,crop))
            
            for t in upInfo['updateParaList']:
                fig = plotParaTimeSeries(para_3I_dic_path='%s/para_3I_%s.pkl'%(outFolder,crop),para=t,FIPS='17001',crop=crop)
                fig.savefig('%s/paraTrend_%s_%s.png'%(outFolder,t,crop))
            GPPmax_3I.to_csv('%s/GPPmax_3I_%s.csv'%(outFolder,crop))
            GPPsum_3I.to_csv('%s/GPPsum_3I_%s.csv'%(outFolder,crop))
