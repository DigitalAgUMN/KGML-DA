# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 16:47:35 2023

@author: yang8460

DA + OP: one-batch(512) takes 16s
         one-batch(1024) takes 27s
"""
import matplotlib.pyplot as plt
from osgeo import gdal
import numpy as np
import pandas as pd
import time
from math import floor
from KGDA_NASApower import NASAPowerWeatherDataProvider
import datetime
import KGDA_util as util
import scipy.signal
import KGDA_Networks as net
import torch
from KFs.ensemble_kalman_filter_torch import EnsembleKalmanFilter_parallel_v4 as EnKF

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device}" " is available.")
def CalRHfromPV(vp,ta):
    """
        
        Parameters
        ----------
        vp : Daily average partial pressure of water vapor. Pa
        es : saturation vapour pressure of water, https://en.wikipedia.org/wiki/Tetens_equation
        ta : air temperature
        Returns
        -------
        None.
    
    """
    es = 0.61078*np.exp((17.27*ta)/(ta + 237.3))
    RH = 100*(vp/1000)/es
    return RH

def daymet2ecosys(daymet,nasa):
    dic = {}
    dic['Tair'] = 0.5*(daymet[:,4] + daymet[:,5])
    dic['RH'] = CalRHfromPV(vp=daymet[:,-1],ta=dic['Tair'])
    dic['Wind'] = np.array(nasa['WS2M'].tolist()).astype(np.float32)
    dic['Precipitation'] = daymet[:,1]
    dic['Radiation'] = daymet[:,0] * daymet[:,2] / 1e6
    return dic

class makeInput_ensemble_parallel():  
    def __init__(self,paraMode,year):
    
        ## in & out
        self.year = year
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
                    if not np.isnan(cropParaCali[0][0,0]):                        
                        for v, vi in zip(cropParaCali[0][0,:],paraList):
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
        
    def disturbPara(self,inputEpisodes, ensemble_n = 100,FIPSbatch=None,disturbPD=False,CV_para=0.1, CV_PD = 0.05,cropTypes=None):
        # np.random.seed(0)
        daily_En_list = []
         
        for inputEpisode,crop in zip(inputEpisodes,cropTypes):
            inputEpisode=np.array(inputEpisode).astype(np.float32)
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
        return daily_En_list   

def loadGPP(obsPathes,obsDateList,DA_period=[151,243], interval=8):
    obsDOY = [t.timetuple().tm_yday for t in obsDateList]
    selected_obsDOY = [t for i,t in enumerate(np.arange(DA_period[0],DA_period[1])) if i%interval==0]
    sliceIndex = [obsDOY.index(t) for t in selected_obsDOY]
    
    # load GPP
    obsImgList = []
    for i,t in enumerate(obsPathes):
        geoimg = gdal.Open(t)
        if i==0:
            gt_forward = geoimg.GetGeoTransform()
            gt_reverse = gdal.InvGeoTransform(gt_forward)
        img = geoimg.ReadAsArray()
        obsImgList.append(img)
        if i%100==0:
            print('%s/%s'%(i+1,len(obsPathes)))
    merge = np.stack(obsImgList)
    
    # filter GPP
    print("Filtering obs...")
    merge = scipy.signal.savgol_filter(merge,5,3,axis=0)
    sliced = merge[sliceIndex,:,:]
    return sliced,gt_forward,gt_reverse,selected_obsDOY

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class ecoNet_env():
    def __init__(self,Q=None):
        self.modelName = 'model/gru-epoch30-batch256-cornBelt20299_4cell_v1_2_RecoCorrected_paraPheno_c2-221010-000709_state_dict.pth'
        input_dim = 21
        output_dim=[1,3,3,2]
        hidden_dim = 64
        mode='paraPheno_c2'
        self.cellRange = [[0,output_dim[0]],
                     [output_dim[0],np.sum(output_dim[0:2])],
                     [np.sum(output_dim[0:2]),np.sum(output_dim[0:3])],
                     [np.sum(output_dim[0:3]),np.sum(output_dim[0:4])]]
        self.model = to_device(net.KG_ecosys(input_dim=input_dim,hidden_dim=hidden_dim,
                                                            output_dim=output_dim,mode=mode), device)
        self.model.load_state_dict(torch.load(self.modelName))     
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
    def __init__(self, obsType, MaskedIndex, stateN):
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
    
def inperiod(d,periodDate):
    if (d>=periodDate[0]) & (d<=periodDate[1]):
        return True
    else:
        return False

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
if __name__ == '__main__':
    # Champaign county location for NASA POWER data
    Lon = -88.2
    Lat = 40.2
    
    # boundary of the demo area
    # leftTop = [662859.26, 1944224.74]
    # rightBottom = [675679.29, 1934393.69]
    
    # v2
    leftTop = [667380.86, 1914362.44]
    rightBottom = [677335.75, 1906916.18]
    
    # Setting
    mode='intevalAcc' # 'global' #
    outPath = 'pixel_simulation/%s_%s'%(datetime.datetime.now().strftime('%Y%m%d_%H%M'),mode)
   
    saveRes = True
    
    DAbatch = 1024#512
    ensemble_n = 100
    CV_para=0.1#0.1
    CV_PD = 0.05
    disturbPD=True    
    X_selectFeatures = ['Tair','RH','Wind','Precipitation','Radiation','GrowingSeason',
                            'CropType','VCMX','CHL4','GROUPX','STMX','GFILL','SLA1','Bulk density','Field capacity'
                            ,'Wilting point','Ks','Sand content','Silt content','SOC','Fertilizer']
    
    for year in [2010,2011,2012,2013,2014]:
        datetime_series = pd.date_range(start='%s-01-01'%year, end='%s-12-31'%year, freq='D')
        weatherFiles = ['E:/My Drive/GEE DaymetV4_Champaign/%s.tif'%t.strftime('%Y%m%d') for t in datetime_series]
        # discard Dec.31 for leap years
        if len(weatherFiles) > 365:
            weatherFiles = weatherFiles[:-1]
            
        ## load the weather files
        # dayl	seconds	0*	86400*	Duration of the daylight period. Based on the period of the day during which the sun is above a hypothetical flat horizon.    
        # prcp	mm	0*	544*	  Daily total precipitation, sum of all forms converted to water-equivalent.    
        # srad	W/m^2	0*	1051*	Incident shortwave radiation flux density, taken as an average over the daylight period of the day.   
        # swe	kg/m^2	0*	13931*	Snow water equivalent, the amount of water contained within the snowpack.   
        # tmax	°C	-60*	60*	Daily maximum 2-meter air temperature. 
        # tmin	°C	-60*	42*	Daily minimum 2-meter air temperature. 
        # vp	Pa	0*	8230*	Daily average partial pressure of water vapor.    
        weatherList = []
        for i,t in enumerate(weatherFiles):
            geoimg = gdal.Open(t)
            if i==0:
                gt_forward_w = geoimg.GetGeoTransform()
                gt_reverse_w = gdal.InvGeoTransform(gt_forward_w)
            img = geoimg.ReadAsArray()
            weatherList.append(img)
            if i%100==0:
                print('%s/%s'%(i+1,len(weatherFiles)))
        weatherMerge = np.stack(weatherList)
        
        ## load NASA power windspeed
        wdp = NASAPowerWeatherDataProvider(latitude=Lat, longitude=Lon, update=False) # a=wdp.df_power
        df_power = wdp.df_power
        period = [datetime.datetime(year,1,1),datetime.datetime(year,12,31)]
        
        ## load gSSURGO data
        gSSURGO_path = 'I:/gSSURGO_data/county_30m'
        soilRasterList = []
        soilProperty_gssurgo = ['BKDS','FC','WP','SCNV','CSAND','CSILT','CORGC']
        soilProperty_ecosys = ['Bulk density','Field capacity','Wilting point','Ks','Sand content','Silt content','SOC']
        for i,t in enumerate(soilProperty_gssurgo):
            geoimg = gdal.Open('%s/FIPS_17019_%s.tif'%(gSSURGO_path,t))
            if i==0:
                gt_forward_s = geoimg.GetGeoTransform()
                gt_reverse_s = gdal.InvGeoTransform(gt_forward_s)
            img = geoimg.ReadAsArray()
            soilRasterList.append(img)
        soilMerge = np.stack(soilRasterList)   
        
        ## load 30m GPP observation
        GPP_path = 'I:/SLOPE_GPP_Champaign_17019/GPP_Daily_georeferenced/%s'%year
        datetime_series_GPP = pd.date_range(start='%s-04-01'%year, end='%s-10-31'%year, freq='D')
        obsPathes = ['%s/GPP_Daily.%s.tif'%(GPP_path,t.strftime('%Y.%m.%d')) for t in datetime_series_GPP]
        obsGPP,gt_forward_o,gt_reverse_o,selected_obsDOY = loadGPP(obsPathes=obsPathes,obsDateList=datetime_series_GPP,DA_period=[151,243], interval=8)
        
        ## load model  
        ecoNet = ecoNet_env()   
        obsMode = 'GPP'
        if obsMode == 'GPP':
            obsType = [2]
            R = [0.02]
            modeList = ['maskc3c4']
            MaskCells = [2,3]
            MaskedIndex = [4,5,6,7,8]
            openLoop = False
            R_adjust = True
        else:
            openLoop = True
            
        # Initialize input classes
        genEn = makeInput_ensemble_parallel(paraMode='default',year=year)
        run = enRun(obsType=obsType,MaskedIndex=MaskedIndex,stateN=9)
        
        # for crop in ['corn','soybean']:
        for crop in ['corn','soybean']:
            ## load county level parameter
            # para = util.loadPara(path='E:/My Drive/PSO_cornBelt',crop=crop,algorithm='PSO',mode='intevalAcc')
            para = util.loadPara(path='E:/My Drive/PSO_cornBelt',crop=crop,algorithm='PSO',mode=mode)
            para_cali = para.getPara(year)
            
            ## split the batch
            points = pd.read_csv(r'F:\MidWest_counties\CDL_points_yearly_Champaign/FIPS_17019_%s_all_pixels_%s.csv'%(crop, year))
            DAbatchList = []
            FIDbatchList = []           
            points_demo = points[(rightBottom[0]>points['x'])&(leftTop[0]<points['x'])&(rightBottom[1]<points['y'])&(leftTop[1]>points['y'])]
            FIDall = list(np.arange(len(points_demo)))
            
            # continue from the breakPoint
            breakPoint = False
            if breakPoint:
                breakPoint_dic = util.load_object('pixel_simulation/res_soybean_2012_breakPoint.pkl')
                unfinishedPoint = [k for k,j in breakPoint_dic['DA'].items() if len(j)==0]
                points_demo = points_demo.iloc[unfinishedPoint[0]:]
                FIDall = unfinishedPoint
                
            for n,p in enumerate(zip(points_demo['x'].tolist(),points_demo['y'])):
                if n == 0:
                    tmp = [p]
                    tmp2 = [FIDall[n]]
                elif n%DAbatch == 0:
                    DAbatchList.append(tmp)
                    FIDbatchList.append(tmp2)
                    tmp = [p]
                    tmp2 = [FIDall[n]]
                else:
                    tmp.append(p)
                    tmp2.append(FIDall[n])
                
            DAbatchList.append(tmp) 
            FIDbatchList.append(tmp2)
            
            ## Run
            # initialized the yield dic        
            if breakPoint:
                yieldTimeSeires = breakPoint_dic['DA']
                yieldTimeSeires_op = breakPoint_dic['op']
            else:
                yieldTimeSeires = {}
                yieldTimeSeires_op = {}
                for t in FIDall:
                    yieldTimeSeires[t] = []
                    yieldTimeSeires_op[t] = []
            eps = 0
            for oneBatch,FID in zip(DAbatchList,FIDbatchList):
                startTime0 = time.time()
                cropTypes = []
                input_df_list = []
                obsList = []
            
                for x,y in oneBatch:
                    # make input
                    input_df = pd.DataFrame(columns=X_selectFeatures)
                    px_w, py_w = gdal.ApplyGeoTransform(gt_reverse_w, x, y)
                    
                    # extract the weather by point
                    daymet = weatherMerge[:,:,floor(py_w),floor(px_w)]
                    
                    # extract windspeed data
                    nasa = wdp.get_period(period)
                    # discard Dec.31 for leap years
                    if len(nasa) > 365:
                        nasa = nasa.iloc[:-1]
                    w_dic = daymet2ecosys(daymet,nasa)
                    for t in ['Tair','RH','Wind','Precipitation','Radiation']:
                        input_df[t] = w_dic[t]
                        
                    # extract soil by point
                    px_s, py_s = gdal.ApplyGeoTransform(gt_reverse_s, x, y)
                    gssurgo = soilMerge[:,floor(py_s),floor(px_s)]
                    for sg,se in zip(gssurgo,soilProperty_ecosys):
                        input_df[se] = sg
                    if crop=='corn':
                        cropTypes.append(0)
                    else:
                        cropTypes.append(1)
                    input_df_list.append(input_df)
                    
                    # extract obs by point
                    px_o, py_o = gdal.ApplyGeoTransform(gt_reverse_o, x, y)
                    measValue = obsGPP[:,floor(py_o),floor(px_o)]
                    if max(measValue) == -9999:
                        obsList.append(None)
                    else:
                        obsList.append(measValue*0.001*genEn.y_NormCoef[obsType[0]])
                
                # fill out the parameters
                para_list = para_cali[['17019']]
                cropParaCali=[np.array(para_list).T,para.paraLoc]
                
                # make ensemble inputs
                inputEpisode_reset = genEn.resetDefaultPara(input_df_list,cropTypes=cropTypes,cropParaCali=cropParaCali)
                enList = genEn.disturbPara(inputEpisode_reset,ensemble_n = ensemble_n,
                                                                          disturbPD=disturbPD,CV_para=CV_para, CV_PD = CV_PD,
                                                                         cropTypes=cropTypes)
                
                # check the validation of data
                valid_enList = []
                valid_obs = []
                valid_FID = []
                for i,j,k in zip(enList,obsList,FID):
                    if j is not None:
                        valid_enList.append(i)
                        valid_obs.append(j)
                        valid_FID.append(k)
                
                # DA
                inputEpisode_en = np.stack(valid_enList)
                measurements = [selected_obsDOY, [valid_obs]]
                measDate = [datetime.datetime(year,1,1)+datetime.timedelta(int(i)-1) for i in selected_obsDOY]
                
                ecoNet.reset()
                xs_enkf,P_enkf,K_enkf,sigmas_enkf = run.oneRun(inputFeature=inputEpisode_en, 
                                                            inputData=genEn, ecoNet=ecoNet, 
                                                            openLoop=openLoop,measurements=measurements,R=R,
                                                            cellRange=ecoNet.cellRange,MaskCells=MaskCells,R_adjust=R_adjust)
                       
                # open-loop
                run_op = enRun(obsType=obsType,MaskedIndex=None,stateN=9)
                ecoNet.reset()
                _,_,_,sigmas_enkf_op = run_op.oneRun(inputFeature=inputEpisode_en, 
                                                            inputData=genEn, ecoNet=ecoNet, 
                                                            openLoop=True,measurements=measurements,R=R,
                                                            cellRange=ecoNet.cellRange,MaskCells=None,R_adjust=R_adjust)
                
                # yield statistics
                _, y_mean_list, _ = ensembleProcess_parallel(sigmas_enkf,i=-1,inputData=genEn)
                _, y_mean_list_op, _ = ensembleProcess_parallel(sigmas_enkf_op,i=-1,inputData=genEn)
                
                # log result
                n = 0
                for s in FID:
                    if s in valid_FID:
                        yieldTimeSeires[s].append(y_mean_list[n][-1])
                        yieldTimeSeires_op[s].append(y_mean_list_op[n][-1])
                        n+=1
                    else:
                        yieldTimeSeires[s].append(None)
                        yieldTimeSeires_op[s].append(None)
                        
                # plot
                if eps<1:       
                    obs_RS = [t/genEn.y_NormCoef[i] for i,t in zip(obsType,np.array(measurements[1][eps]).astype(np.float32))]
                    obs_date_RS = measDate
                    
                    if 'GPP' in obsMode.split('_'):
                        loc=obsMode.split('_').index('GPP')
                        i=2                               
                        fig1 = visualizeResult(i=i,predictions_all=sigmas_enkf[0,:,:,:],predictions_op=sigmas_enkf_op[0,:,:,:],
                                               inputData=genEn,periodDate=None,mode='RSobs'
                                        ,obs_RS=obs_RS[loc],obs_date_RS=obs_date_RS,yearS=year,obsName='SLOPE GPP')
                    else:
                        i=2
                        fig1 = visualizeResult(i=i,predictions_all=sigmas_enkf[0,:,:,:],predictions_op=sigmas_enkf_op[0,:,:,:]
                                               ,inputData=genEn,periodDate=None,yearS=year)
                
                    i=-1
                    fig4 = visualizeResult(i=i,predictions_all=sigmas_enkf[0,:,:,:],predictions_op=sigmas_enkf_op[0,:,:,:]
                                           ,inputData=genEn,periodDate=None,yearS=year)
                eps+=1    
                finishTime0 = time.time()
                if eps%10==0:
                    print('%s of %s, Batch %s, take %.2f s for one-batch, %s/%s pixels finished.'%(crop,year,eps,(finishTime0-startTime0),eps*DAbatch,len(points_demo)))
                
            # save results
            if saveRes:
                util.mkdir(outPath)
                resDic={}
                resDic['DA'] = yieldTimeSeires
                resDic['op'] = yieldTimeSeires_op
                util.save_object(resDic,'%s/res_demoArea_%s_%s.pkl'%(outPath,crop,year))
        
        ## save break point
        # resDic={}
        # resDic['DA'] = yieldTimeSeires
        # resDic['op'] = yieldTimeSeires_op
        # util.save_object(resDic,'%s/res_%s_%s_breakPoint.pkl'%(outPath,crop,year))
        
        # break
    # plt.figure()
    # plt.scatter(w_dic['RH'], nasa['RH'].tolist())
    
    # plt.figure()
    # plt.scatter(w_dic['Tair'], nasa['T2M'].tolist())
    
    # plt.figure()
    # plt.scatter(w_dic['Precipitation'], nasa['PRECTOTCORR'].tolist())
    
    # plt.figure()
    # plt.scatter(w_dic['Radiation'], nasa['ALLSKY_SFC_SW_DWN'].tolist())
