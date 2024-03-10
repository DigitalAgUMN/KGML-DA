# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 13:00:02 2022

@author: yang8460

v1，0: cut the computational cost from 116s to 44s per county (11 sites)
v1.2: Torch version of EnKF, 44s to 25s， ps: cov() and inv() still need loop
v1.3: Torch version of EnKF, 25s to 14s， ps: inv() still need loop
v2: Torch version of EnKF, 14s to 12s， ps: All are batch modes
v3: enable sites batch mode
v3.1: screen the purity of sites
v3.2: use corrected GPP, disturb planting date & add process error; remove the PD disturb in parareset phase
v3.3: calibration-updating rotation strategy, 2022-9-21
v3.4: add the std adaption strategy
v3.5: extend study area to whole cornbelt with new model
v3.6: assimilate LAI and ET
v3.7: combination of obs, batch mode
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
from KFs.ensemble_kalman_filter_torch import EnsembleKalmanFilter_parallel_v4 as EnKF
import scipy.signal
import ECONET_Test_county_plot as plot
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['figure.dpi'] = 300

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device}" " is available.")

def classifyPointsByCounty(siteFile):
    ## create county-based list
    site_info = pd.read_csv(siteFile)
    FIPSList = list(set(site_info.FIPS[~np.isnan(site_info.FIPS)].astype(int).tolist()))
    FIPSList.sort()
    
    ## group points
    FIPS_dic = {}
    for t in FIPSList:
        tmp=site_info[site_info.FIPS==t]
        FIPS_dic[t]=tmp['Site_ID'].tolist()
    return FIPS_dic,FIPSList,site_info

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
        self.cornFixPloc = [7]
        self.soybeanFixPloc = [8,-1]
        self.gounpXloc = 9
        self.gounpXup = 21
        
            
        self.X_selectFeatures = ['Tair','RH','Wind','Precipitation','Radiation','GrowingSeason',
                            'CropType','VCMX','CHL4','GROUPX','STMX','GFILL','SLA1','Bulk density','Field capacity'
                            ,'Wilting point','Ks','Sand content','Silt content','SOC','Fertilizer']   
        self.plantingDate_corn = [5,1] #  planting date
        self.plantingDate_soybean = [5,20]
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
        
    def disturbPara(self,inputEpisodes, ensemble_n = 100,FIPSbatch=None,disturbPD=False,CV_para=0.1, CV_PD = 0.05,obs_smooth_dic_all=None,
                        measDOY=None,year=None,obsType=None,startYear=2000,cropTypes=None):
        # np.random.seed(0)
        daily_En_list = []
        validFIPS = []
        measValue = []
         
        for inputEpisode,FIPS,crop in zip(inputEpisodes,FIPSbatch,cropTypes):
            obsv = [t[FIPS][year-startYear] is None for t in obs_smooth_dic_all]
            if (inputEpisode is None)|max(obsv):
                continue
            validFIPS.append(FIPS)
            inputEpisode=np.array(inputEpisode).astype(np.float32)
            tmp=[]
            for i,obsT in enumerate(obsType):
                tmp.append(obsExtrat(obs_smooth_dic_all[i],FIPS,measDOY,year,
                                           coef=self.y_NormCoef[obsT],startYear=startYear))
            measValue.append(tmp)
            # disturb parameters
            u_para_default = inputEpisode[0,7:]
            CV = [CV_para]*len(u_para_default)
            # CV = [0]*17 # Coefficient of Variation
            std2_0 = list((np.asarray(u_para_default) * np.asarray(CV))**2)
            P_u_para = np.diag(std2_0)
            ensemble_paras = list(np.random.multivariate_normal(mean=u_para_default, cov=P_u_para, size=ensemble_n))
            
            # disturb planting date
            pD = list(inputEpisode[:,5]).index(1)+1
            ensemble_PD = list(np.random.normal(loc=pD,scale=(CV_PD*pD), size=ensemble_n))
            daily_in_ensemble = []
            
            # reconstruct the input
            if crop == 0:
                fixP = self.cornFixPloc
                self.xl = np.array([100,0.02,15,2,0.0003,0.005,1.5])
                self.xu = np.array([150,0.07,21,8,0.0007,0.025,8])
                self.paraLoc = [5,8,9,10,11,12,20]
            else:
                fixP = self.soybeanFixPloc
                self.xl = np.array([120,20,16,2,0.0003,0.005])
                self.xu = np.array([170,70,21,6,0.0007,0.015])
                self.paraLoc = [5,7,9,10,11,12]
                
            for i in range(ensemble_n):
                tmp = inputEpisode.copy()
                for n,p in enumerate(ensemble_paras[i]):                   
                    if (n+7) in fixP:
                        continue
                    
                    # constrain the gounpx if it is outbound
                    # if (n+7) == self.gounpXloc:
                    #     if p > self.gounpXup:
                    #         continue
                    if (n+7) in self.paraLoc:
                        loc = self.paraLoc.index(n+7)
                        if (p>self.xu[loc]) | (p<self.xl[loc]):
                            continue
                    tmp[:,n+7] = p
                                
                # growing season
                season = np.zeros((tmp.shape[0])).astype(np.float32)
                season[int(ensemble_PD[i])-1:] = 1
                tmp[:,5] = season
                
                daily_in_ensemble.append(tmp)
            
            daily_in_ensemble_array = np.array(daily_in_ensemble)
            daily_En_list.append(daily_in_ensemble_array)
        return daily_En_list, validFIPS, measValue

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class ecoNet_env():
    def __init__(self,model, Q=None):
        self.model = model
        self.reset()
        self.Q = Q
        if Q is not None:
            print ("Process error is %.5f"%Q)
        
    def reset(self):
        self.hidden_state = None
        self.previousLAI = None
        
    def steprun(self, x, updateState=None):
        '''
        
        Parameters
        ----------
        x : shape:[sites,ensemble,dt,features]
        
        Q : process error, added by Qi 2022-9-30

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
        
        if self.Q is not None:
            # e = multivariate_normal(self._mean, self.Q, N)  # discard by Qi 2020/8/31
            # self.sigmas += e
            noise = torch.tensor(np.random.multivariate_normal(mean=np.zeros(yhat.shape[-1]), 
                                                               cov=self.Q*np.eye(yhat.shape[-1]),
                                                               size=xCompact.shape[0]).astype(np.float32)[:,np.newaxis,:]).to(device)
            yhat += noise    
        self.previousLAI = yhat[:,0,LAIloc].view([-1,1])

        # decompact batch
        yhat_decompact = torch.split(yhat,self.enN,dim=0)
        return yhat_decompact

class enRun():
    def __init__(self, obsType=None, MaskedIndex=None, stateN=None):
        self.obsType=obsType
        self.MaskedIndex=MaskedIndex
        self.stateN=stateN
        if len(self.obsType) >0:
            tmp = []
            for t in self.obsType:
                tmp.append(torch.unsqueeze(torch.zeros(self.stateN, dtype=torch.float32),dim=1).to(device))
                tmp[-1][t] = 1
            self.H = torch.concat(tmp,dim=1)
        
    def oneRun(self, inputFeature,inputData, ecoNet, openLoop=False, measurements =None, R=None,cellRange=None,MaskCells=None,R_adjust=False):
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
            return None,None,None,predictions_decompact
        else:
    
            ecoNet.model.eval()
            
            # assimilation
            self.siteN = inputFeature.shape[0]
            ensemble_n = inputFeature.shape[1]
            P0 = np.diag([0.])
            DA_enkf = EnKF(x=np.zeros(len(inputData.y_selectFeatures)), P=P0, 
                           dim_z=len(self.obsType), N=ensemble_n, H = self.H, fx = ecoNet.steprun,cellRange=cellRange)
            DA_enkf.R = np.diag([t**2 for t in R])
            
            xs_enkf = []
            sigmas_enkf = []
            P_enkf = []
            K_enkf = []
    
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
                
                ## predict
                DA_enkf.predict(dailyIn,updateHidden,MaskCells =MaskCells)
                updateHidden = False
                
                ## update
                if DAP in z_DAP:
                    # print('debug,update phase, DAP is %s'%DAP)
                    # adaption the std if the RE is too high
                    obsAll = [[np.array([0.0]) if z[i] is None else np.array([z[i][ass_n]]) for z in zs] for i in range(len(self.obsType))]                    
                    MaskedSample = [i for i,z in enumerate(zs) if z is None]
                    if len(MaskedSample) ==0:
                        MaskedSample = None
                        
                    if R_adjust:
                        # v_pre  = np.squeeze(DA_enkf.Pz_batch.cpu().detach().numpy())
                        v_pre  = DA_enkf.Pz_batch.cpu().detach().numpy()
                        RR = []
                        for j in range(len(self.obsType)):                            
                            RR.append(np.squeeze(self.R_adjust(xs_enkf,index=j,obs=obsAll[j],std_pre=np.sqrt(v_pre[:,j,j]),R=R[j])))
                        RR = np.stack(RR).T
                        RR = np.stack([np.diag(t) for t in RR])
                    else:
                        RR=None
                    zList = np.concatenate([np.array(t) for t in obsAll],axis=1)
                    DA_enkf.update(zList=zList,MaskedIndex=self.MaskedIndex,R=RR,MaskedSample=MaskedSample)
                    updateHidden = True
                    ass_n += 1
                    K_enkf.append(DA_enkf.K_batch.cpu().detach().numpy().copy())
                  
                ## record
                xs_enkf.append(DA_enkf.x_batch.cpu().detach().numpy().copy())
                P_enkf.append(DA_enkf.P_batch.cpu().detach().numpy().copy())   
                sigmas_enkf.append(DA_enkf.sigmas_batch.cpu().detach().numpy().copy())
                
                DAP += 1
            if self.siteN == 1:
                sigmas_enkf = np.squeeze(np.array(sigmas_enkf)).transpose(1,0,2)[np.newaxis,:,:,:]
            else:
                sigmas_enkf = np.squeeze(np.array(sigmas_enkf)).transpose(1,2,0,3)
            return xs_enkf,P_enkf,K_enkf,sigmas_enkf

    def R_adjust(self,xs_enkf,index,obs,std_pre,R):
        obs_pre = xs_enkf[-1][:,self.obsType[index]]                       
        RE_obs = np.abs(obs_pre-np.squeeze(np.array(obs)))/obs_pre                    
        R1 = std_adjust(RE=RE_obs, std_pre=std_pre, std_obs=R)**2
        RR = R1[:,np.newaxis,np.newaxis]
        return RR
    
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
    
def ensembleProcess_parallel(predictions,i,inputData):
    y_en_list = []
    y_mean_list = []
    y_std_list = []
    for t in list(predictions):
        tmp = t[:,:,i]/inputData.y_NormCoef[i]
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

def ensembleProcess(predictions,i,inputData):
    t=predictions

    tmp = t[:,:,i]/inputData.y_NormCoef[i]
    y_en = (tmp)
    y_mean = np.mean(tmp,axis=0)
    y_std = np.std(tmp,axis=0)

    return y_en, y_mean, y_std  

def visualizeResult(i,predictions_all,predictions_op,inputData,periodDate,mode='noObs',obs_RS=None,obs_date_RS=None, yearS = 2000,title='',obsName=''):

    y_en, y_mean, y_std = ensembleProcess(predictions_all,i,inputData)
    dateList = [datetime.datetime(yearS,1,1)+datetime.timedelta(days=t) for t in range(len(y_mean))]
    
    # if mode == 'noObs':      
    #     fig=plot_test_series(sList=[y_mean,y_en],labelList=['Data Assimilation Mean','Data Assimilation Ensemble'],fmtList = ['line','ensemble'],
    #                             ylabel=inputData.y_selectFeatures[i],title=title,dateCol =[dateList,dateList],periodDate=periodDate)

    # elif mode == 'RSobs':
    #     fig=plot_test_series(sList=[y_mean,y_en, obs_RS],
    #                      labelList=['Data Assimilation Mean','Data Assimilation Ensemble','SLOPE GPP'],fmtList = ['line','ensemble','marker_circle'],
    #                             ylabel=inputData.y_selectFeatures[i],title=title,dateCol =[dateList,dateList,obs_date_RS],periodDate=periodDate)
    
    y_en_op, y_mean_op, y_std_op = ensembleProcess(predictions_op,i,inputData) 
    if mode == 'noObs':      
        fig=plot_test_series(sList=[y_mean_op,y_en_op,y_mean,y_en],labelList=['Without DA','Open-loop Ensemble','Data Assimilation (DA)','Data Assimilation Ensemble'],
                             fmtList = ['line','ensemble','line','ensemble'],
                                ylabel=inputData.y_selectFeatures[i],title=title,dateCol =[dateList,dateList,dateList,dateList],periodDate=periodDate)

    elif mode == 'RSobs':
        fig=plot_test_series(sList=[y_mean_op,y_en_op,y_mean,y_en, obs_RS],
                         labelList=['Without DA','Open-loop Ensemble','Data Assimilation (DA)','Data Assimilation Ensemble',obsName],
                         fmtList = ['line','ensemble','line','ensemble','marker_circle'],
                                ylabel=inputData.y_selectFeatures[i],title=title,dateCol =[dateList,dateList,dateList,dateList,obs_date_RS],periodDate=periodDate)
    return fig


def appendYield(yieldTimeSeires,yieldYear,index):
    if len(yieldYear) == 0:
        yieldTimeSeires[index].append(np.nan)
    else:
        yieldTimeSeires[index].append(np.mean(yieldYear))
    return yieldTimeSeires

def obsExtrat(obs_smooth_dic,FIPS,measDOY,year,coef,startYear=2000):
    obs = obs_smooth_dic[FIPS][year-startYear]
    if obs is not None:
        return [obs[t-1]*coef for t in measDOY]
    else:
        return None

def returnMax(std1,std2):
    tmp = std2.copy()
    tmp[tmp<std1] = std1
    return tmp
    
def std_adjust(RE, std_pre, std_obs, ratio1 = 1.0,ratio2 = 1.5):
    
    new_std = np.ones(len(RE))*std_obs
    
    # RE loc
    loc1 = np.where((RE>0.3)&(RE<=0.5))
    loc2 = np.where(RE>0.5)
    
    # 0.3<RE<0.5
    ratio1_pre = ratio1*std_pre
    tmp = returnMax(std1=std_obs, std2=ratio1_pre)
    new_std[loc1] = tmp[loc1]
    
    # RE>0.5
    ratio2_pre = ratio2*std_pre
    tmp = returnMax(std1=std_obs, std2=ratio2_pre)
    new_std[loc2] = tmp[loc2]
    
    return new_std   

def paddingObs(data):
    doy_modis = [i for i in range(1,365, 8)]
    doy_data = doy_modis[-len(data):]
    intact = np.ones(365)*np.nan
    intact[list(np.array(doy_data)+4-1)] = data
    return intact

def returnObs(yearRange,tmp):
    obs = []
    for t in range(yearRange[0],yearRange[1]+1):
        if tmp['data'][t] is None:
            obs.append(None)
        else:
            if np.isnan(tmp['data'][t][0]):
                obs.append(None)
            else:
                obs.append(tmp['data'][t])
    return obs

def loadStat(OBSpath,term,mode='max'):
    RS_OBS = {}
    for crop in ['corn','soybean']:
        RS_OBS[crop]= pd.read_csv('%s/%s_%s_%s.csv'%(OBSpath,term,mode,crop))
        # RS_OBS[crop].insert(0,'Year',yearRange_t)
    return RS_OBS

def adaptCoef(obs_dic, outVarIndex=-2,mode='max',globalf = False):
    # output stat
    outVarPath = 'F:/MidWest_counties/STAT/outVars_case_%s_cornbelt/trend_var_%s.pkl'%(paraMode,mode)
    outDic = util.load_object(outVarPath)
    _=outDic['corn'].pop('Year')
    _=outDic['soy'].pop('Year')
    mean_out_dic = {}
    for t in ['corn','soy']:
        mean_out_dic[t] = np.mean(np.stack([np.array(t) for t in outDic[t].values()]).transpose((1,0,2)),axis=1)
        
    factor = {}
    for crop in ['corn','soy']:
        mean_obs = np.nanmean(np.array(obs_dic['soybean' if crop== 'soy' else crop]),axis=1)
        mean_out = mean_out_dic[crop][:,outVarIndex]
        if globalf:
            factor[crop] = np.ones(len(mean_obs)) * np.mean(mean_out)/np.mean(mean_obs)
        else:
            factor[crop] = mean_out/mean_obs
        
    return factor
    
if __name__ == '__main__':
    
    yearRange = [2000,2020]
    yearRange_t = [t for t in range(yearRange[0],yearRange[1]+1)]
    periodDate=[datetime.datetime.strptime('%s0101'%(yearRange[0]),'%Y%m%d'),
                                              datetime.datetime.strptime('%s1231'%(yearRange[1]),'%Y%m%d')]
    yearSpan = np.arange(yearRange[0],yearRange[1]+1)
    
    # load county-level sites
    NASS_Path = 'F:/MidWest_counties/Yield'

    # obs type
   
    saveFig = True
    saveRes = True
    disturbPD = True
    GPPpath = 'F:/MidWest_counties/GPP'
    LAIpath = 'F:/MidWest_counties/GLASS_LAI_v2'
    ETpath = 'F:/MidWest_counties/MODIS_ET_v2'
    globalf=True
    
    for paraMode in ['default','intevalAcc']:#'default''defaultV2''intevalAcc'#,'eachYear'#'previousYear'#'inteval'#'global'#'eachYear'#'intevalAcc' 
    
        for obsMode in ['','GPP','GPP_ET','GPP_LAI_ET']:
        
            case='obs_%s_globalf'%(obsMode)
            if obsMode == 'GPP':
                obsType = [2]
                R = [0.02]
                modeList = ['maskc3c4']
                
            elif obsMode == 'ET':
                obsType = [1]
                R = [0.16]
                modeList = ['maskc3c4']
                
            elif obsMode == 'LAI':
                obsType = [-2]
                R = [0.02]
                # R = [0.02]
                modeList = ['maskc2c3']
                
            elif obsMode == 'GPP_ET':
                obsType = [2,1]
                R = [0.02,0.08]
                modeList = ['maskc3c4']
                
            elif obsMode == 'GPP_LAI':
                obsType = [2,-2]
                R = [0.02,0.02]
            elif obsMode == 'GPP_LAI_ET':
                obsType = [2,-2,1] # GPP,LAI,ET
                R = [0.02,0.02,0.08]#np.diag([0.02**2])  # measurement cov matrix
                modeList = ['maskc3']
                # modeList = ['maskc3yield']
            elif obsMode == '':
                modeList = ['openLoop']
                R=None
                obsType = []
                
            CV_para=0.1#0.1
            CV_PD = 0.05
            
            dataRoot = r'F:/MidWest_counties/inputMerged_DA_countyMerge'
            countyPathes = glob.glob('%s/*.pkl'%dataRoot)
            FIPSList_all = [t.split('\\')[-1].split('_')[0] for t in countyPathes]
            
            # pick valid FIPS
            FIPSList = []
            for f in FIPSList_all:
                if os.path.exists('%s/GPP_%s_corn.pkl'%(GPPpath,f)) & os.path.exists('%s/GPP_%s_soybean.pkl'%(GPPpath,f)) & \
                   os.path.exists('%s/GLASS_LAI_%s_corn.pkl'%(LAIpath,f)) & os.path.exists('%s/GLASS_LAI_%s_soybean.pkl'%(LAIpath,f)) & \
                       os.path.exists('%s/MODIS_ET_%s_corn.pkl'%(ETpath,f)) & os.path.exists('%s/MODIS_ET_%s_soybean.pkl'%(ETpath,f)):
                    FIPSList.append(f)
            
            now = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
            
            R_adjust = True#True
            
         
            ensemble_n = 100#100
            # note = 'county_yield_parallel_25296'
            obsIntertval = 8
            
            modelVersion = [
                            ['epoch30-batch256','model/gru-epoch30-batch256-cornBelt20299_4cell_v1_2_RecoCorrected_paraPheno_c2-221010-000709_state_dict.pth']
                            ] 
            
            # para info
            algorithm = 'PSO'
            paraPath = 'H:/My Drive/PSO_cornBelt'
            
            
            # modeList = ['maskNone','maskc3c4','maskc1c3c4','maskc4','maskc1','maskYield']
            # modeList = ['maskc3c4']
            # modeList = ['maskLAI']
            # modeList = ['maskc3yield']
            # modeList = ['maskc1c3c4']
            # modeList = ['maskc1yield']
            # modeList = ['maskNone','maskYield','maskc1','maskc1c2c3','maskc1c2c3yield']
            # modeList = ['openLoop','maskc2c3yield']
            
            
            # load networks
            mode='paraPheno_c2'
            batchFIPS = 320
            input_dim = 21
            output_dim=[1,3,3,2]
            hidden_dim = 64 # 64
            cellRange = [[0,output_dim[0]],
                         [output_dim[0],np.sum(output_dim[0:2])],
                         [np.sum(output_dim[0:2]),np.sum(output_dim[0:3])],
                         [np.sum(output_dim[0:3]),np.sum(output_dim[0:4])]]
            
            ## obs bias correction
            # obs stat
            GPPall = loadStat(GPPpath,term='GPP')
            LAIall = loadStat(LAIpath,term='LAI')
            ETall = loadStat(ETpath,term='ET',mode='MeanGS')
             
            correction_coef_LAI = adaptCoef(obs_dic=LAIall, outVarIndex=-2,globalf=globalf)
            correction_coef_ET = adaptCoef(obs_dic=ETall, outVarIndex=1,mode='MeanGS',globalf=globalf)
                                                          
            for modelv in modelVersion:
                for maskMode in modeList:
                    note = 'county_yield_parallel_%s_%s_obsInterval%s_countyMerge_%s_case%s'%(modelv[0],maskMode,obsIntertval,paraMode,case)
                    projectName = '%s-%s'%(note,now)
                    outFolder = 'F:/countyYieldResult/EnsembleSigma/%s'%(projectName)
                    
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
                 
                    model = to_device(net.KG_ecosys(input_dim=input_dim,hidden_dim=hidden_dim,
                                                                output_dim=output_dim,mode=mode), device)
        
                    modelName = modelv[1]
                    model.load_state_dict(torch.load(modelName))     
                    ecoNet = ecoNet_env(model)
                    
                    run = enRun(obsType=obsType,MaskedIndex=MaskedIndex,stateN=9)
        
                    yield_3I_all_dic = {}
                    
                    # for crop in ['soy']:
                    for crop in ['corn','soy']:
                       
                        # load para 
                        para = util.loadPara(paraPath,crop,algorithm,mode=paraMode)
                        paraLoc = para.paraLoc
                        
                        # run model for each county
                        yield_3I_dic = {}
                        yield_3I_dic['Year'] = yearSpan
                        yield_3I_pom_dic = {}
                        yield_3I_pom_dic['Year'] = yearSpan
                        
                        eps = 0
                        # measDOY = [i for i in range(140,280,10)]
                        # measDOY = [i for i in range(130,280, obsIntertval)]
                        # measDOY = [i for i in range(160, 220, obsIntertval)]  # Jun.1 = Sep.1
                        measDOY = [i for i in range(145+4,270, obsIntertval)] # + 4 because the modis 8days composite product is compositing following 8 days
                        
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
                        
                        count = 0
          
                        for batch,FIPSbatch in enumerate(FIPSbatchList):
                            inputMerged_site = []
                            obs_smooth_dic = {}
                            obs_smooth_dic2 = {}
                            obs_smooth_dic3 = {}
                            startTime = time.time()
        
                            for FIPS in FIPSbatch:                          
                                # load site data
                                tmp = []
                                tmp2 = []
                                tmp3 = []
                                
                                # load input and Ecosys output data
                                inputDataPath = '%s/%s_inputMerged.pkl'%(dataRoot,FIPS)
                                inputData = util.load_object(inputDataPath)
                                inputMerged_site.append(inputData)
                        
                                # fetch and process obs data
           
                                if crop=='corn':
                                    if ('GPP' in obsMode.split('_')) | (obsMode == ''):
                                        tmp = util.load_object('%s/GPP_%s_%s.pkl'%(GPPpath,FIPS,crop))
                                    if 'LAI' in obsMode.split('_'):
                                        tmp2 = util.load_object('%s/GLASS_LAI_%s_%s.pkl'%(LAIpath,FIPS,crop))
                                    if 'ET' in obsMode.split('_'):
                                        tmp3 = util.load_object('%s/MODIS_ET_%s_%s.pkl'%(ETpath,FIPS,crop))
                                else:
                                    if ('GPP' in obsMode.split('_')) | (obsMode == ''):
                                        tmp = util.load_object('%s/GPP_%s_%sbean.pkl'%(GPPpath,FIPS,crop))
                                    if 'LAI' in obsMode.split('_'):
                                        tmp2 = util.load_object('%s/GLASS_LAI_%s_%sbean.pkl'%(LAIpath,FIPS,crop))
                                    if 'ET' in obsMode.split('_'):
                                        tmp3 = util.load_object('%s/MODIS_ET_%s_%sbean.pkl'%(ETpath,FIPS,crop))
                                if ('GPP' in obsMode.split('_')) | (obsMode == ''):
                                    obs = [None if np.isnan(tmp[t][0]) else tmp[t] for t in range(yearRange[0],yearRange[1]+1)]
                                    obs_smooth_dic[FIPS] = [scipy.signal.savgol_filter(t,5,3) if t is not None else None for t in obs]  # GPP
                                if 'LAI' in obsMode.split('_'):
                                    obs2 = returnObs(yearRange,tmp2)
                                    obs_smooth_dic2[FIPS] = [paddingObs(scipy.signal.savgol_filter(t,5,3))*correction_coef_LAI[crop][y-2000] \
                                                             if t is not None else None for t,y in zip(obs2,yearRange_t)]  # LAI
                                if 'ET' in obsMode.split('_'):
                                    obs3 = returnObs(yearRange,tmp3)
                                    obs_smooth_dic3[FIPS] = [paddingObs(scipy.signal.savgol_filter(t,5,3))*correction_coef_ET[crop][y-2000] \
                                                             if t is not None else None for t,y in zip(obs3,yearRange_t)]  #  ET
                                                    
                                
                            genEn = makeInput_ensemble_parallel(inputMerged=inputMerged_site,paraMode=paraMode)
                    
                            measList = []
                            measDateList = []
                            yieldTimeSeires = {}
                            yieldTimeSeires_std = {}
                            
                            if (obsMode == 'GPP') | (obsMode == ''):
                                obs_smooth_dic_all=[obs_smooth_dic]
                            elif obsMode == 'ET':
                                obs_smooth_dic_all=[obs_smooth_dic3]
                            elif obsMode == 'LAI':
                                obs_smooth_dic_all=[obs_smooth_dic2]
                            elif obsMode == 'GPP_ET':
                                obs_smooth_dic_all=[obs_smooth_dic,obs_smooth_dic3]
                            elif obsMode == 'GPP_LAI':
                                obs_smooth_dic_all=[obs_smooth_dic,obs_smooth_dic2]
                            elif obsMode == 'GPP_LAI_ET':
                                obs_smooth_dic_all=[obs_smooth_dic,obs_smooth_dic2,obs_smooth_dic3]
                         
                            
                            for year in range(yearRange[0],yearRange[1]+1):
                                
                                # load para 
                                para_cali = para.getPara(year)
                                if para_cali is None:
                                    cropParaCali=None
                                else:
                                    para_list = para_cali[FIPSbatch]
                                    cropParaCali=[np.array(para_list).T,paraLoc]
                                                      
                                if crop=='corn':
                                    cropTypes = [0 for _ in range(len(FIPSbatch))]
                                else:
                                    cropTypes = [1 for _ in range(len(FIPSbatch))]
                                inputEpisodes = genEn.get(year)
                                inputEpisode_reset = genEn.resetDefaultPara(inputEpisodes,cropTypes=cropTypes,cropParaCali=cropParaCali)
                                
                                enList, validFIPS, measValue = genEn.disturbPara(inputEpisode_reset,ensemble_n = ensemble_n,FIPSbatch=FIPSbatch,
                                                                      disturbPD=disturbPD,CV_para=CV_para, CV_PD = CV_PD,
                                                                      obs_smooth_dic_all=obs_smooth_dic_all,
                                                                      measDOY=measDOY,year=year,obsType=obsType,cropTypes=cropTypes,startYear=yearRange[0])
                                inputEpisode_en = np.stack(enList)
                                ecoNet.reset()
                                measDate = [datetime.datetime(year,1,1)+datetime.timedelta(i-1) for i in measDOY]
                              
            
                                measurements = [measDOY, measValue]
            
                                xs_enkf,P_enkf,K_enkf,sigmas_enkf = run.oneRun(inputFeature=inputEpisode_en, 
                                                                            inputData=genEn, ecoNet=ecoNet, 
                                                                            openLoop=openLoop,measurements=measurements,R=R,
                                                                            cellRange=cellRange,MaskCells=MaskCells,R_adjust=R_adjust)
                                
                                
                                # sigmasList.append([sigmas_enkf,validFIPS])
                                if measurements is not None:
                                    measList.extend(measurements[1])
                                    measDateList.extend([datetime.datetime.strptime('%d%03d'%(year,t),'%Y%j') for t in measurements[0]])
                                
                                # yield statistics
                                _, y_mean_list, y_std_list = ensembleProcess_parallel(sigmas_enkf,i=-1,inputData=genEn)
                             
                    
                                if eps<1:
                                    # open-loop
                                    run_op = enRun(obsType=obsType,MaskedIndex=None,stateN=9)
                                    ecoNet.reset()
                                    _,_,_,sigmas_enkf_op = run_op.oneRun(inputFeature=inputEpisode_en, 
                                                                                inputData=genEn, ecoNet=ecoNet, 
                                                                                openLoop=True,measurements=measurements,R=R,
                                                                                cellRange=cellRange,MaskCells=None,R_adjust=R_adjust)
                                    
                                    obs_RS = [t/genEn.y_NormCoef[i] for i,t in zip(obsType,np.array(measurements[1][eps]).astype(np.float32))]
                                    obs_date_RS = measDate
                                    
                                    if 'GPP' in obsMode.split('_'):
                                        loc=obsMode.split('_').index('GPP')
                                        i=2                               
                                        fig1 = visualizeResult(i=i,predictions_all=sigmas_enkf[0,:,:,:],predictions_op=sigmas_enkf_op[0,:,:,:],
                                                               inputData=genEn,periodDate=periodDate,mode='RSobs'
                                                        ,obs_RS=obs_RS[loc],obs_date_RS=obs_date_RS,yearS=year,obsName='SLOPE GPP')
                                    else:
                                        i=2
                                        fig1 = visualizeResult(i=i,predictions_all=sigmas_enkf[0,:,:,:],predictions_op=sigmas_enkf_op[0,:,:,:]
                                                               ,inputData=genEn,periodDate=periodDate,yearS=year)
                                    i=4
                                    fig2 = visualizeResult(i=i,predictions_all=sigmas_enkf[0,:,:,:],predictions_op=sigmas_enkf_op[0,:,:,:]
                                                           ,inputData=genEn,periodDate=periodDate,yearS=year)
                                   
                                    
                                    if 'LAI' in obsMode.split('_'):
                                        loc=obsMode.split('_').index('LAI')
                                        i=-2
                                        fig3 = visualizeResult(i=i,predictions_all=sigmas_enkf[0,:,:,:],predictions_op=sigmas_enkf_op[0,:,:,:]
                                                               ,inputData=genEn,periodDate=periodDate,mode='RSobs'
                                                               ,obs_RS=obs_RS[loc],obs_date_RS=obs_date_RS,yearS=year,obsName='GLASS LAI')
                                    else:
                                        i=-2
                                        fig3 = visualizeResult(i=i,predictions_all=sigmas_enkf[0,:,:,:],predictions_op=sigmas_enkf_op[0,:,:,:]
                                                               ,inputData=genEn,periodDate=periodDate,yearS=year)
                                        
                                    i=-1
                                    fig4 = visualizeResult(i=i,predictions_all=sigmas_enkf[0,:,:,:],predictions_op=sigmas_enkf_op[0,:,:,:]
                                                           ,inputData=genEn,periodDate=periodDate,yearS=year)
                                    
                                    if 'ET' in obsMode.split('_'):
                                        loc=obsMode.split('_').index('ET')
                                        i=1
                                        fig5 = visualizeResult(i=i,predictions_all=sigmas_enkf[0,:,:,:],predictions_op=sigmas_enkf_op[0,:,:,:]
                                                               ,inputData=genEn,periodDate=periodDate,mode='RSobs'
                                                               ,obs_RS=obs_RS[loc],obs_date_RS=obs_date_RS,yearS=year,obsName='MODIS ET')
                                    else:
                                        i=1
                                        fig5 = visualizeResult(i=i,predictions_all=sigmas_enkf[0,:,:,:],predictions_op=sigmas_enkf_op[0,:,:,:]
                                                               ,inputData=genEn,periodDate=periodDate,yearS=year)
                                    
                                    i=0
                                    fig6 = visualizeResult(i=i,predictions_all=sigmas_enkf[0,:,:,:],predictions_op=sigmas_enkf_op[0,:,:,:]
                                                           ,inputData=genEn,periodDate=periodDate,yearS=year)
                                    
                                    i=3
                                    fig7 = visualizeResult(i=i,predictions_all=sigmas_enkf[0,:,:,:],predictions_op=sigmas_enkf_op[0,:,:,:]
                                                           ,inputData=genEn,periodDate=periodDate,yearS=year)
                                    
                                    if saveRes:
                                        if not os.path.exists(outFolder):
                                            os.makedirs(outFolder)
                                        fig1.savefig('%s/test_scatter_GPP_%s_%s_%s.png'%(outFolder,eps,crop,FIPS))
                                        fig2.savefig('%s/test_scatter_biomass_%s_%s_%s.png'%(outFolder,eps,crop,FIPS))
                                        fig3.savefig('%s/test_scatter_LAI_%s_%s_%s.png'%(outFolder,eps,crop,FIPS))
                                        fig4.savefig('%s/test_scatter_Yield_%s_%s_%s.png'%(outFolder,eps,crop,FIPS))
                                        fig5.savefig('%s/test_scatter_ET_%s_%s_%s.png'%(outFolder,eps,crop,FIPS))
                                        fig6.savefig('%s/test_scatter_DVS_%s_%s_%s.png'%(outFolder,eps,crop,FIPS))
                                        fig7.savefig('%s/test_scatter_SM_%s_%s_%s.png'%(outFolder,eps,crop,FIPS))
                                eps+=1
                                
                                # classify the FIPS         
                                if year == yearRange[0]:
                                    n = 0
                                    for s in FIPSbatch:
                                        if s in validFIPS:
                                            yieldTimeSeires[s] = [y_mean_list[n][-1]]
                                            yieldTimeSeires_std[s] = [y_std_list[n][-1]]
                                            
                                            n+=1
                                        else:
                                            yieldTimeSeires[s] = [None]
                                            yieldTimeSeires_std[s] = [None]
                                           
                                else:
                                    n = 0
                                    for s in FIPSbatch:
                                        if s in validFIPS:
                                            yieldTimeSeires[s].append(y_mean_list[n][-1])
                                            yieldTimeSeires_std[s].append(y_std_list[n][-1])
                                           
                                            n+=1
                                        else:
                                            yieldTimeSeires[s].append(None)
                                            yieldTimeSeires_std[s].append(None)
                                         
                                                   
                            for n,t in enumerate(FIPSbatch):            
                                yield_3I_dic[t] = yieldTimeSeires[t]
                                yield_3I_pom_dic[t] = yieldTimeSeires_std[t]
                                finishTime = time.time()
                            count+= len(FIPSbatch)
                            print('%d/%d counties finished, take %.2f s.'%(count,len(FIPSList),finishTime-startTime))
                            # if saveRes:
                            #     E_data.save_object(sigmasList,'%s/sigmasLog_%s.pkl'%(outFolder,batch))
                                
                        # save result
                        yield_3I = pd.DataFrame(yield_3I_dic)  
                        yield_3I_pom = pd.DataFrame(yield_3I_pom_dic)  
                        yield_3I_all_dic[crop] = yield_3I
                        if saveRes:
                            
                            if crop=='corn':
                                yield_3I.to_csv('%s/yield_%s.csv'%(outFolder,crop))
                                yield_3I_pom.to_csv('%s/yield_pom_%s.csv'%(outFolder,crop))
                            else:
                                yield_3I.to_csv('%s/yield_soybean.csv'%(outFolder))
                                yield_3I_pom.to_csv('%s/yield_pom_soybean.csv'%(outFolder))
                                
                     
                    summary_corn, summary_soybean = plot.NASS_vs_EcoNet_cornBelt(NASS_Path,NetResult=outFolder,saveFig=saveFig,outFolder=outFolder)
                    # summary_corn, summary_soybean = plot.NASS_vs_EcoNet_cornBelt(NASS_Path = 'E:/NASS_yield',NetResult=outFolder,
                    #                                                              saveFig=False,outFolder=outFolder)
                    
                    # correct yield trend
                    for crop in ['corn','soybean']:
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
                        yield_df, yield_df_origin = util.individualTrend(df_yield_nass_,outFolder,crop,mode=paraMode,interval=3,
                                                                         yearRange_t=yearRange_t,coef=coef)
                    
                        # show trend corrected discard first three years and some nan couties
                        validFIPS_GPP,_ = util.validGPPcounty()
                        validFIPS_GPP_ = list(set(validFIPS_GPP).intersection(set(yield_df.columns)))
                        #  = 
                        util.NASS_vs_ecosys(df_NASS=df_yield_nass_.iloc[3:],df_p=yield_df[['Year']+validFIPS_GPP_].iloc[3:],crop=crop,
                                       outFolder=outFolder,mode='trendCorrect_From2003_%s'%paraMode,saveFig=saveFig)
                    