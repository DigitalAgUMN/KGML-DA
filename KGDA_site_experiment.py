# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 15:42:35 2024

@author: yang8460

Data assimilation for Ne-1,2,3 sites
"""

import os
import numpy as np
import time
import torch
import KGDA_Networks as net
import pandas as pd
import datetime
from KGDA_NASApower import NASAPowerWeatherDataProvider
import KGDA_siteData as E_site
import matplotlib.pyplot as plt
import matplotlib
from KFs.ensemble_kalman_filter_torch import EnsembleKalmanFilter_parallel_v4 as EnKF
import scipy.signal

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['figure.dpi'] = 300

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device}" " is available.")
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def loadPara(filename,smethod='space'):
    with open(filename, 'r') as f:
            data = f.readlines()
    if smethod=='space':
        data = [t.strip().split() for t in data]
    elif smethod=='comma':
        data = [t.strip().split(',') for t in data]
    return data

def hx(x):
    return np.array([x[2]])
        
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
        if len(inputFeature.shape)==3:
            inputFeature = np.expand_dims(inputFeature, axis=0)
        if openLoop:
            
            ecoNet.model.eval()
            yhat_list = []
            for i in range(inputFeature.shape[2]):
                dailyIn = inputFeature[:,:,i,:][:,:,np.newaxis,:].astype(np.float32)   
                
                ## predict
                yhat = torch.stack(ecoNet.steprun(dailyIn))
                yhat_list.append(yhat.detach().cpu().numpy())
            
            # decompact batch
            predictions_decompact = np.squeeze(np.stack(yhat_list),axis=3).transpose(1,2,0,3)#         
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
                            tmp = np.squeeze(self.R_adjust(xs_enkf,index=j,obs=obsAll[j],std_pre=np.sqrt(v_pre[:,j,j]),R=R[j]))
                            if len(tmp.shape)==0:
                                tmp = np.expand_dims(tmp,axis=0)
                            RR.append(tmp)
                            # RR.append(self.R_adjust(xs_enkf,index=j,obs=obsAll[j],std_pre=np.sqrt(v_pre[:,j,j]),R=R[j]))
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

def returnMax(std1,std2):
    tmp = std2.copy()
    tmp[tmp<std1] = std1
    return tmp

class makeInput_ensemble():  
    def __init__(self, test_site, wdp, fert=None):
        
        self.test_site=test_site
        self.wdp=wdp
        self.site_planting_date = pd.DataFrame(columns=['Year','US-Ne1','US-Ne2','US-Ne3'])
        self.site_planting_date['Year'] = np.arange(2001,2008+1)
        self.site_planting_date['US-Ne1'] = [130,129,135,126,124,125,121,120]
        self.site_planting_date['US-Ne2'] = [131,140,134,154,123,132,122,136]
        self.site_planting_date['US-Ne3'] = [134,140,133,155,117,131,121,135]
        
        self.site_crop = pd.DataFrame(columns=['Year','US-Ne1','US-Ne2','US-Ne3'])
        self.site_crop['Year'] = np.arange(2001,2008+1)
        self.site_crop['US-Ne1'] = [0,0,0,0,0,0,0,0]  # all maize
        self.site_crop['US-Ne2'] = [0,1,0,1,0,1,0,1]  # maize-soybean rotation
        self.site_crop['US-Ne3'] = [0,1,0,1,0,1,0,1]  # maize-soybean rotation rainfed
        
        self.site_yield = pd.DataFrame(columns=['Year','US-Ne2','US-Ne3'])
        self.site_yield['Year'] = np.arange(2001,2008+1)
        self.site_yield['US-Ne1'] = [14.29,13.24,12.32,12.45,12.63,11.79,13.1,14.65]  # maize ton/ha
        self.site_yield['US-Ne2'] = [13.41,3.99,14,3.71,13.24,4.36,13.21,4.22]  # maize-soybean rotation ton/ha
        self.site_yield['US-Ne3'] = [8.72,3.32,7.72,3.41,9.1,4.31,10.23,3.97] # maize-soybean rotation rainfed
        
        ## in & out
        self.cropParaList = ['VCMX','CHL4','GROUPX','STMX','GFILL','SLA1']
        self.cropPara_index = [7,8,9,10,11,12]
        # self.cropParaDefaults = [[90, 0.05, 19, 5, 0.0005, 0.018],
        #                     [45, 0.0, 17, 4, 0.0005, 0.008]]  # 0:maize,  1:soybean
        self.cropParaDefaults = [[90, 0.05, 17, 5, 0.0005, 0.019],
                            [45, 0.0, 18, 4, 0.0005, 0.008]]  # 0:maize,  1:soybean
        
        # self.cropParaDefaults = [[90, 0.10, 23, 6, 0.0005, 0.015],
        #                           [20, 0.0, 23, 4, 0.0005, 0.005]]  # 0:maize,  1:soybean
        
        # self.y_selectFeatures = ['DVS','GPP_daily','Biomass','LAI','GrainYield']
        self.y_selectFeatures = ['DVS',
                            'ET_daily','GPP_daily','AVE_SM',
                            'Biomass','Reco_daily','NEE_daily',
                            'LAI','GrainYield',]
        self.y_NormCoef = [0.5,
                      0.15, 0.02, 1.5,
                      0.001, 0.06, -0.05,
                      0.1,0.0015]
        
        self.fert = fert
        self.X_selectFeatures = ['Tair','RH','Wind','Precipitation','Radiation','GrowingSeason',
                                'CropType','VCMX','CHL4','GROUPX','STMX','GFILL','SLA1','Bulkdensity','Fieldcapacity'
                                ,'Wiltingpoint','Ks','Sandcontent','Siltcontent','SOC','Fertilizer']
 
  
        # soil para
        data = loadPara('demoData/siteData/gSSURGO_Ecosys/mesoil_site%s'%(test_site+1), smethod='comma')
        # soil data, average of 0-n layer, that is, 0.01 - 0.3
        self.soilParaList = [('Bulk density',2,[0,3]),('Field capacity',3,[0,3]),
                        ('Wilting point',4,[0,3]),('Ks',5,[0,3]),('Sand content',7,[0,3]),
                        ('Silt content',8,[0,3]),('SOC',14,[0,3])]
        self.soilPara_index = [13,14,15,16,17,18,19]
        self.df_s = pd.DataFrame(columns=[t[0] for t in self.soilParaList])
        self.df_s.loc[0] = [np.mean(np.array(data[t[1]][t[2][0]:t[2][1]]).astype(np.float32)) for t in self.soilParaList]
        
        # try-error
        # self.df_s['Sand content'] = 600
        
        # weather from wdp
        self.var_climate = ['T2M','RH','WS2M','PRECTOTCORR','ALLSKY_SFC_SW_DWN']
        self.var_climate_index = [0,1,2,3,4]
    
    def coef_C2dryMatter(self, cropType):
        #  dry matter soybean contains 54% carbon, maize contains 45% carbon. g C/m2 to ton dry matter / m2
        if cropType == 0: #Maize
            return 0.01/0.45
        else: # soybean
            return 0.01/0.54
        
    def get(self, year, DOY):
        siteName = site_info.iloc[test_site]['Site_ID']
        plantingDate = int(self.site_planting_date.loc[self.site_planting_date[self.site_planting_date['Year']==year].index][siteName])
        cropType = int(self.site_crop.loc[self.site_crop[self.site_crop['Year']==year].index][siteName])
    
        ## make daily input
        daily_in = np.ones(len(self.X_selectFeatures))*-1
        
        # feed weather data
        date = datetime.datetime(year,1,1)+datetime.timedelta(days=DOY-1)
        weather = self.wdp.get_daily(date)
        for v, vi in zip(self.var_climate,self.var_climate_index):
            daily_in[vi] = weather.iloc[0][v]
            
        # feed growing season & crop type
        if DOY >= plantingDate:
            daily_in[5] = 1
        else:
            daily_in[5] = 0
        
        daily_in[6] = cropType
        
        # feed crop para
        for v, vi in zip(self.cropParaDefaults[cropType],self.cropPara_index):
            daily_in[vi] = v
        
        # feed soil para
        for v, vi in zip(self.soilParaList,self.soilPara_index):
            daily_in[vi] = self.df_s.iloc[0][v[0]]
        
        # fert
        daily_in[-1] = self.fert
        
        daily_in = daily_in[np.newaxis,:]
        
        return daily_in

    def disturbPara(self,inputEpisode, ensemble_n = 100, CV_PD = 0.05):
        np.random.seed(0)
        u_para_default = inputEpisode[0,7:]
        CV = [0.1]*len(u_para_default)
        # CV = [0]*17 # Coefficient of Variation
        std2_0 = list((np.asarray(u_para_default) * np.asarray(CV))**2)
        P_u_para = np.diag(std2_0)
        ensemble_paras = list(np.random.multivariate_normal(mean=u_para_default, cov=P_u_para, size=ensemble_n))
        
        # disturb planting date
        pD = list(inputEpisode[:,5]).index(1)+1
        ensemble_PD = list(np.random.normal(loc=pD,scale=(CV_PD*pD), size=ensemble_n))
            
        daily_in_ensemble = []
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
        return daily_in_ensemble_array
        
def inperiod(d,periodDate):
    if (d>=periodDate[0]) & (d<=periodDate[1]):
        return True
    else:
        return False
    
def plot_test_series(sList,labelList,fmtList,outFolder=None,saveFig=False,note='',title='',dateCol = None, scale = 1.0, periodDate=None,
                     color_list = None):
    
    sList=[np.array(t).astype(float) for t in sList]
       
    # fig = plt.figure(figsize=(13,5))
    # fig, ax = plt.subplots(figsize=(5,3.5))
    fig, ax = plt.subplots(figsize=(6,3.5))
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
                plt.plot_date(x,s*scale, color=color_list[i], linestyle='-' ,fmt='None',alpha=1,linewidth=0.8)
            elif fmtList[i] =='marker':   
                plt.plot_date(x,s*scale, color=color_list[i],  label=labelList[i], fmt='*',markersize=4, alpha=.6)
            elif fmtList[i] =='marker_circle':   
                plt.plot_date(x,s*scale, color=color_list[i],  fmt='o', alpha=1,markersize=8,fillstyle='none')
            elif fmtList[i] =='ensemble':
                for t in range(s.shape[0]):
                    if t==0:
                        plt.plot_date(x,s[t,:],color = color_list[i] , linestyle='-', fmt='None', alpha=.05,linewidth=0.5)#, label=labelList[i])
                    else:
                        plt.plot_date(x,s[t,:],color = color_list[i] , linestyle='-', fmt='None', alpha=.05,linewidth=0.5)
    plt.legend(fontsize = 12,frameon=False,loc="upper right",ncol=2)
    plt.xlabel('Timeline',fontsize = 16)
    if title == 'GPP_daily':
        title = 'GPP gC/m2'
    elif title == 'Reco_daily':
        title = 'Reco gC/m2'
    elif title == 'Biomass':
        title = 'Biomass gC/m2'
    elif title == 'LAI':
        title = 'LAI m2/m2'
        
    plt.ylabel(title,fontsize = 16)
    # plt.ylim([-4,40]) 
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

def ensembleProcess(predictions_all,i,inputData):
    y_mean = []
    y_std = []
    y_en = []
    # for t in np.squeeze(predictions_all):
    for t in predictions_all:
        t = np.squeeze(t)
        tmp = t[:,:,i]/inputData.y_NormCoef[i]
        y_en.append(tmp)
        y_mean.append(np.mean(tmp,axis=0))
        y_std.append(np.std(tmp,axis=0))
    y_en = np.concatenate(y_en,axis=1)
    y_mean = np.concatenate(y_mean,axis=0) 
    y_std = np.concatenate(y_std,axis=0)
    return y_en, y_mean, y_std  

def plotScatterDense(x_, y_,alpha=1, binN=200, thresh_p = 0.05,outFolder='',
                     saveFig=False,note='',title='',uplim=None,downlim=None,auxText = None,legendLoc=4):
        fig, ax = plt.subplots(1, 1,figsize = (7,5))
        x_=np.array(x_)
        y_=np.array(y_)
        if len(y_) > 1:
                       
            # Calculate the point density
            if not (thresh_p is None):
                thresh = (np.max(np.abs(x_))*thresh_p)
                x = x_[(x_>thresh)|(x_<-thresh)]
                y = y_[(x_>thresh)|(x_<-thresh)]
            else:
                x = x_
                y = y_
                
            para = np.polyfit(x, y, 1)
            y_fit = np.polyval(para, x)  #
            plt.plot(x, y_fit, 'r')
        
        #histogram definition
        bins = [binN, binN] # number of bins
        
        # histogram the data
        hh, locx, locy = np.histogram2d(x, y, bins=bins)

        # Sort the points by density, so that the densest points are plotted last
        z = np.array([hh[np.argmax(a<=locx[1:]),np.argmax(b<=locy[1:])] for a,b in zip(x,y)])
        idx = z.argsort()
        x2, y2, z2 = x[idx], y[idx], z[idx]

        plt.scatter(x2, y2, c=z2, cmap='jet', marker='.',alpha=alpha)
        
        if uplim==None:
            uplim = 1.2*max(np.hstack((x, y)))
        if downlim==None:
            downlim = 0.8*min(np.hstack((x, y)))
            
        figRange = uplim - downlim
        plt.plot(np.arange(downlim,np.ceil(uplim)+1), np.arange(downlim,np.ceil(uplim)+1), 'k', label='1:1 line')
        plt.xlim([downlim, uplim])
        plt.ylim([downlim, uplim])
        plt.xlabel('Observations',fontsize=16)
        plt.ylabel('Predictions',fontsize=16)
        # plt.legend(loc=1) 
        if not legendLoc is None:
            if legendLoc==False:
                plt.legend(edgecolor = 'w',facecolor='w',fontsize=12)          
            else:
                plt.legend(loc = legendLoc, edgecolor = 'w',facecolor='w',fontsize=12, framealpha=0)
        plt.title(title, y=0.9)
        
        if len(y) > 1:
            R2 = np.corrcoef(x, y)[0, 1] ** 2
            RMSE = (np.sum((y - x) ** 2) / len(y)) ** 0.5
            # MAPE = 100 * np.sum(np.abs((y - x) / (x+0.00001))) / len(x)
            plt.text(downlim + 0.1 * figRange, downlim + 0.83 * figRange, r'$R^2 $= ' + str(R2)[:5], fontsize=14)
            # ax = plt.text(0.1 * uplim, 0.86 * uplim, r'$MAPE $= ' + str(MAPE)[:5] + '%', fontsize=14)
            plt.text(downlim + 0.1 * figRange, downlim + 0.76 * figRange, r'$RMSE $= ' + str(RMSE)[:5], fontsize=14)
        
            plt.text(downlim + 0.1 * figRange, downlim + 0.69 * figRange, r'$Slope $= ' + str(para[0])[:5], fontsize=14)
        if not auxText == None:
            plt.text(0.05, 0.9, auxText, transform=ax.transAxes,fontproperties = 'Times New Roman',fontsize=20)
        plt.colorbar()
        
        if saveFig:
            plt.title('%s samples all'%title)
            fig.savefig('%s/test_scatter_%s.png'%(outFolder,note))

        else:
            plt.title(title)
            
def visualizeResult(i,predictions_all,predictions_all_op,inputData,periodDate,mode='noObs',obs=None,obs_date=None,obs_RS=None,obs_date_RS=None,dense=False):

    y_en_op, y_mean_op, y_std_op = ensembleProcess(predictions_all_op,i,inputData)
    y_en, y_mean, y_std = ensembleProcess(predictions_all,i,inputData)
    dateList = [datetime.datetime(yearRange[0],1,1)+datetime.timedelta(days=t) for t in range(len(y_mean))]
    
    if mode == 'noObs':
        color_list = ['darkgreen','g','grey','darkred','r','m','b','k','y','c','sienna','navy','grey']
        plot_test_series(sList=[y_mean,y_en,y_mean_op, y_en_op],labelList=['Data Assimilation (DA)','Data Assimilation Ensemble','Without DA','Open-loop Ensemble'],fmtList = ['line','ensemble','line','ensemble'],
                                title=inputData.y_selectFeatures[i],dateCol =[dateList,dateList,dateList,dateList],
                                periodDate=periodDate,color_list=color_list)
    elif mode == 'siteObs':
        color_list = ['darkgreen','g','grey','darkred','r','m','b','k','y','c','sienna','navy','grey']
        # plot_test_series(sList=[y_mean,y_en,y_mean_op,y_en_op,obs],labelList=['Data Assimilation (DA)','Data Assimilation Ensemble','Without DA','Open-loop Ensemble','Ground truth'],
        #                  fmtList = ['line','ensemble','line','ensemble','marker'],
        #                         title=inputData.y_selectFeatures[i],dateCol =[dateList,dateList,dateList,dateList,obs_date],periodDate=periodDate)
        plot_test_series(sList=[y_mean_op,y_en_op,obs,y_mean,y_en],labelList=['Without DA','Open-loop Ensemble',ylabel,'Data Assimilation (DA)','Data Assimilation Ensemble'],
                         fmtList = ['line','ensemble','marker','line','ensemble'],
                                title=inputData.y_selectFeatures[i],dateCol =[dateList,dateList,obs_date,dateList,dateList],
                                periodDate=periodDate,color_list=color_list)
    elif mode == 'siteRSobs':
        color_list = ['darkgreen','g','grey','darkred','r','m','b','k','y','c','sienna','navy','grey']
        # plot_test_series(sList=[y_mean,y_en,y_mean_op,y_en_op,obs,obs_RS],
        #                  labelList=['Data Assimilation (DA)','Data Assimilation Ensemble','Without DA','Open-loop Ensemble','Ground truth','Remote sensing GPP'],fmtList = ['line','ensemble','line','ensemble','marker','marker_circle'],
        #                         title=inputData.y_selectFeatures[i],dateCol =[dateList,dateList,dateList,dateList,obs_date,obs_date_RS],
        #                         periodDate=periodDate,color_list=color_list)
        plot_test_series(sList=[y_mean_op,y_en_op,obs,y_mean,y_en,obs_RS],
                         labelList=['Without DA','Open-loop Ensemble','Flux tower GPP','Data Assimilation (DA)','Data Assimilation Ensemble','Remote sensing GPP'],
                         fmtList = ['line','ensemble','marker','line','ensemble','marker_circle'],
                                title=inputData.y_selectFeatures[i],dateCol =[dateList,dateList,obs_date,dateList,dateList,obs_date_RS],
                                periodDate=periodDate,color_list=color_list)
        
    if dense:
        obs = np.array(obs)
        obs_date = np.array(obs_date)
        loc = np.where(np.isnan(obs) == False)
        obs = obs[loc]
        obs_date = obs_date[loc]
        tmp = [(t,t2) for t,t2 in zip(obs_date,obs) if inperiod(t, [dateList[0],dateList[-1]])]
        obs_date = np.array([t[0] for t in tmp])
        obs = np.array([t[1] for t in tmp])        
        pre = [p for p,t in zip(y_mean,dateList) if t in obs_date]
        pre_op = [p for p,t in zip(y_mean_op,dateList) if t in obs_date]
        plotScatterDense(x_=obs, y_=pre, title='%s_after_assimilation'%inputData.y_selectFeatures[i],thresh_p=None)
        plotScatterDense(x_=obs, y_=pre_op, title='%s_open_loop_runs'%inputData.y_selectFeatures[i],thresh_p=None)

def printTime(timeNode,note=''):
    current = time.time()  
    print('Process %s takes %.2f s'%(note,current-timeNode))
    return current

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
    
    ## site info
    test_site = 1
    # for k in range(2007,2008):
    for k in range(2002,2003):    
        yearRange = [k,k]
        periodDate=[datetime.datetime.strptime('%s0401'%(yearRange[0]),'%Y%m%d'),
                                                  datetime.datetime.strptime('%s1031'%(yearRange[1]),'%Y%m%d')]
       
        site_info = pd.DataFrame(columns=['Site_ID','Lon','Lat'])
        site_info['Site_ID'] = ['US-Ne1','US-Ne2','US-Ne3']
        site_info['Lon'] = [-96.4766,-96.4701,-96.4397]
        site_info['Lat'] = [41.1651,41.1649,41.1797]
    
        siteName = site_info.iloc[test_site]['Site_ID']
        fert=6
        ensemble_n = 100
    
        # daily weather data
        lon,lat = [site_info.iloc[test_site]['Lon'], site_info.iloc[test_site]['Lat']]
        wdp = NASAPowerWeatherDataProvider(latitude=lat, longitude=lon, update=False) # a=wdp.df_power
        inputData = makeInput_ensemble(test_site, wdp, fert=fert)
        
        # build model, load parameters
        input_dim = len(inputData.X_selectFeatures)
        output_dim=[1,3,3,2]
        hidden_dim = 64 # 64
        cellRange = [[0,output_dim[0]],
                     [output_dim[0],np.sum(output_dim[0:2])],
                     [np.sum(output_dim[0:2]),np.sum(output_dim[0:3])],
                     [np.sum(output_dim[0:3]),np.sum(output_dim[0:4])]]
        R_adjust = False#True
        R = [0.01]
        
        # load observation data
        LAI_df_list,biomass_df_list,LMA_df_list,prod_df_list,planting_df_list,harvest_df_list = \
            E_site.read_BIF_id(siteList=[siteName])
        GPP_SLOPE, LAI_RS = E_site.read_RS_product()
        
        startTime = time.time()

        # load a trained network
        model = net.KG_ecosys(input_dim=input_dim,hidden_dim=hidden_dim,output_dim=output_dim)
        model = to_device(model, device)  
        modelName ='demo_model_cornBelt20299_state_dict.pth'
        model.load_state_dict(torch.load(modelName))    
        
        ecoNet = ecoNet_env(model)
        timeNode = time.time()
        
        obsType = [2]
        MaskCells,MaskedIndex,openLoop = maskinfo(maskMode='maskc3c4')
        # MaskCells = [3]
        # MaskCells,MaskedIndex,openLoop = maskinfo(maskMode='maskc4')
        
        measDOY = [i for i in range(150,280,10)]
        
        # run open-loop
        run_op = enRun(obsType=obsType,MaskedIndex=None,stateN=9)
        predictions_all_op = []
        for year in range(yearRange[0],yearRange[1]+1):
            inputEpisode = np.concatenate([inputData.get(year, DOY=t) for t in range(1,365)],axis=0)
            if test_site !=2:
                # added some rain
                for d in range(inputEpisode.shape[0]):
                    if d in measDOY:
                        inputEpisode[d,3] += 60
            inputEpisode_en = inputData.disturbPara(inputEpisode, ensemble_n = ensemble_n)
    
            ecoNet.reset()
            _,_,_,sigmas_enkf_op = run_op.oneRun(inputFeature=inputEpisode_en, 
                                                        inputData=inputData, ecoNet=ecoNet, 
                                                        openLoop=True,
                                                        cellRange=cellRange,MaskCells=None,R_adjust=R_adjust)
           
    
            predictions_all_op.append(sigmas_enkf_op)
        timeNode = printTime(timeNode,note='open-loop run')
         
        # run ensemble
        run = enRun(obsType=obsType,MaskedIndex=MaskedIndex,stateN=9)
        predictions_all = []
        input_all = []
        GPP_smooth = pd.DataFrame(columns=['Date',siteName])
        GPP_smooth['Date'] = GPP_SLOPE['Date'].tolist()
        GPP_smooth[siteName] = scipy.signal.savgol_filter(GPP_SLOPE[siteName].tolist(),5,3)
        
    
        measList = []
        measDateList = []
        # R = np.diag([0.02**2])  # measurement cov matrix
        
        for year in range(yearRange[0],yearRange[1]+1):
            inputEpisode = np.concatenate([inputData.get(year, DOY=t) for t in range(1,365)],axis=0)
            if test_site !=2:
                # added some rain
                for d in range(inputEpisode.shape[0]):
                    if d in measDOY:
                        inputEpisode[d,3] += 60
            inputEpisode_en = inputData.disturbPara(inputEpisode, ensemble_n = ensemble_n)
            
            measDate = [datetime.datetime(year,1,1)+datetime.timedelta(i-1) for i in measDOY]
            measValue = [GPP_smooth[siteName][GPP_smooth['Date']==t].values[0]*inputData.y_NormCoef[2] for t in measDate]
            ecoNet.reset()
            measurements = [measDOY, [[measValue]]]
            # measurements = None
            xs_enkf,P_enkf,K_enkf,sigmas_enkf = run.oneRun(inputFeature=inputEpisode_en, 
                                                        inputData=inputData, ecoNet=ecoNet, 
                                                        openLoop=openLoop,measurements=measurements,R=R,
                                                        cellRange=cellRange,MaskCells=MaskCells,R_adjust=R_adjust)
            
            predictions_all.append(sigmas_enkf)
            input_all.append(inputEpisode_en)
            if measurements is not None:
                measList.extend(measurements[1][0][0])
                measDateList.extend([datetime.datetime.strptime('%d%03d'%(year,t),'%Y%j') for t in measurements[0]])
        
        timeNode = printTime(timeNode,note='EnKF run') 
        
        period = ['%d0101'%yearRange[0], '%d1231'%yearRange[1]]    
        NEE_data,GPP_data,Reco_data,energyBalance,ET_data,SWC_data,SWC_data_10cm_AVE,SWC_data_30cm_AVE = E_site.read_flux(siteList=[siteName], period=period)
        
        # DVS
        i=0
        ylabel='DVS'
        visualizeResult(i,predictions_all,predictions_all_op,inputData,mode='noObs',periodDate=periodDate)
        
        # GPP
        i=2
        ylabel='GPP'
        obs = np.array(GPP_data[siteName]).astype(np.float32)
        obs_date = pd.to_datetime(GPP_data['Date'].astype(str)).tolist()
        obs_RS = np.array(measList).astype(np.float32)/inputData.y_NormCoef[i]
        obs_date_RS = measDateList
        visualizeResult(i,predictions_all,predictions_all_op,inputData,mode='siteRSobs',obs=obs,obs_date=obs_date,
                        obs_RS=obs_RS,obs_date_RS=obs_date_RS,periodDate=periodDate)

        # biomass
        i=4
        ylabel='Ground truth of aboveground biomass'
        obs = np.array(biomass_df_list[0]['AG_BIOMASS_CROP']).astype(np.float32)
        obs_date = biomass_df_list[0]['AG_BIOMASS_DATE'].tolist()
        visualizeResult(i,predictions_all,predictions_all_op,inputData,mode='siteObs',obs=obs,obs_date=obs_date,periodDate=periodDate)
    
        # Reco
        i=5
        ylabel='Flux tower Reco'
        obs = np.array(Reco_data[siteName]).astype(np.float32)
        obs_date = pd.to_datetime(Reco_data['Date'].astype(str)).tolist()
        # visualizeResult(i,predictions_all,predictions_all_op,inputData,mode='siteObs',obs=obs,obs_date=obs_date,periodDate=periodDate,dense=True)
        visualizeResult(i,predictions_all,predictions_all_op,inputData,mode='siteObs',obs=obs,obs_date=obs_date,periodDate=periodDate)
        # visualizeResult(i,predictions_all,predictions_all_op,inputData,mode='siteObs',obs=obs,obs_date=obs_date,periodDate=periodDetail)
        
        # NEE
        i=6
        ylabel='Flux tower NEE'
        obs = np.array(NEE_data[siteName]).astype(np.float32)
        obs_date = pd.to_datetime(NEE_data['Date'].astype(str)).tolist()
        visualizeResult(i,predictions_all,predictions_all_op,inputData,mode='siteObs',obs=obs,obs_date=obs_date,periodDate=periodDate)
      
        
        # LAI
        i=7
        ylabel='Ground truth of LAI'
        obs = np.array(LAI_df_list[0]['LAI_TOT']).astype(np.float32)
        obs_date = LAI_df_list[0]['LAI_DATE'].tolist()
        visualizeResult(i,predictions_all,predictions_all_op,inputData,mode='siteObs',obs=obs,obs_date=obs_date,periodDate=periodDate)
           

