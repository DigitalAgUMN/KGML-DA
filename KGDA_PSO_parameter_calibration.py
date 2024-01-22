# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 15:43:20 2023

@author: yang8460
"""

from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.problems.single import Rastrigin
from pymoo.optimize import minimize
import pymoo.gradient.toolbox as anp
import numpy as np
from pymoo.core.problem import Problem
import matplotlib.pyplot as plt
import KGDA_Networks as net
import torch
import datetime
import os, glob
import scipy.signal
import pandas as pd
# from pymoo.core.problem import ElementwiseProblem
from pymoo.termination import get_termination
import matplotlib
import sys
version = int(sys.version.split()[0].split('.')[1])
if version > 7:
    import pickle
else:
    import pickle5 as pickle
    
device = "cuda" if torch.cuda.is_available() else "cpu"

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
        
def load_object(filename):
    with open(filename, 'rb') as inp:
        data = pickle.load(inp)
    return data

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
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
    def resetDefaultPara(self,inputEpisodes,cropTypes):
        inputEpisode_reset_list = []
        for inputEpisode,cropType in zip(inputEpisodes,cropTypes):
            if inputEpisode is None:
                inputEpisode_reset_list.append(None)
            else:
                inputEpisode_reset = inputEpisode.copy()
                inputEpisode_reset['CropType'] = cropType
                for v, vi in zip(self.cropParaDefaults[cropType],self.cropParaList):
                    inputEpisode_reset[vi] = v       
                if cropType==0:
                    inputEpisode_reset['Fertilizer'] = self.fert
                    dateP = datetime.date(self.year,self.plantingDate_corn[0],self.plantingDate_corn[1])
                else:
                    inputEpisode_reset['Fertilizer'] = 0
                    dateP = datetime.date(self.year,self.plantingDate_soybean[0],self.plantingDate_soybean[1])
                          
                # set growing season          
                dateP_DOY = dateP.timetuple().tm_yday
                tmp = np.array(inputEpisode_reset['GrowingSeason'].copy())
                tmp[dateP_DOY-1:] = 1
                inputEpisode_reset['GrowingSeason'] = tmp
                inputEpisode_reset_list.append(inputEpisode_reset)
        return inputEpisode_reset_list

class ECONET():
    def __init__(self):
        input_dim = 21
        output_dim=[1,3,3,2]
        hidden_dim = 64
        mode='paraPheno_c2'        
        self.model = to_device(net.KG_ecosys(input_dim=input_dim,hidden_dim=hidden_dim,
                                                    output_dim=output_dim,mode=mode), device)
        modelName ='demo_model_cornBelt20299_state_dict.pth'
        self.model.load_state_dict(torch.load(modelName))
        
    def run(self,x):
        self.model.eval()
        out,_ = self.model(torch.tensor(x.astype(np.float32)).to(device))
        return out.detach().cpu().numpy()

   
class myProblem(Problem):
    def __init__(self, crop='corn',model=None, inputTemplet=None,
                 obsYield=None,obsGPP=None,yearRange=None,obsType=None):
        if crop=='corn':
            n_var = 7
            xl = np.array([100,0.02,15,2,0.0003,0.005,1.5])
            xu = np.array([150,0.07,21,8,0.0007,0.025,8])
            self.paraLoc = [5,8,9,10,11,12,20]
        elif crop=='soy':
            n_var = 6
            xl = np.array([120,20,16,2,0.0003,0.005])
            xu = np.array([170,70,21,8,0.0007,0.015])
            self.paraLoc = [5,7,9,10,11,12]
        super().__init__(n_var=n_var, n_obj=1, xl=xl, xu=xu, vtype=float)
        # self.model = model
        self.inputData = inputTemplet
        self.obsYield = obsYield
        self.obsGPP = np.stack(obsGPP)
        self.yearRange = yearRange
        self.obsType = obsType # yield
        self.daySeason = [31,30,31,31,30] # days for May, Jun, Jul, Aug and Sep [31,30,31,31,30]
    
    def divideMonth(self,series):
        out = []
        start = 0
        for d in self.daySeason:
            out.append(np.sum(series[:,start:start+d],axis=1))
            start += d

        return np.array(out).T.reshape(-1)

    def replaceInpput(self,x):
        inputSample_en = []
        for xs in list(x):
            length = self.inputData.shape[1]
            para = np.array([t for t in xs]*length).reshape(length,-1)
            # planting data
            DOY_p = int(xs[0])
            tmp = np.zeros((length))
            tmp[DOY_p-1:,] = 1
            para[:,0] = tmp
            inputSample = self.inputData.copy()
            inputSample[:,:,self.paraLoc] = para
            inputSample_en.append(inputSample)
            
        return inputSample_en
    
    def simulation(self,x):        
        #Here the model is actualy startet with one paramter combination
        self.enN = x.shape[0]
        xList = self.replaceInpput(x)
        xCompact = np.concatenate(xList,axis=0)
        out = model.run(xCompact)
        # decompact batch
        out_decom = np.split(out,self.enN,axis=0)
        sim_opt1,sim_opt2,sim_opt3 = self.splitOut(out_decom)
        return sim_opt1,sim_opt2,sim_opt3
    
    def splitOut(self,outList):
        sim_opt1 = [np.squeeze(out[:,:,self.obsType[0]])[:,-2] for out in outList]  # yield
        sim_opt2 = [np.sum(np.squeeze(out[:,:,self.obsType[1]])[:,120:273],axis=1) for out in outList]  # GPP, sum of May.1 - Otc.1
        sim_opt3 = [self.divideMonth(np.squeeze(out[:,:,self.obsType[1]])[:,120:273]) for out in outList]  # monthly GPP, May.1 - Otc.1,
        return sim_opt1,sim_opt2,sim_opt3
    
    def rmse(self,eva, sim):
        if len(eva) == len(sim) > 0:
            obs, sim = np.array(eva), np.array(sim)
            mse = np.nanmean((obs - sim)**2,axis=1)
            return np.sqrt(mse)
        else:
            print("evaluation and simulation lists do not have the same length.")
            return np.nan
    
    def decomposeObs(self):
        return np.array(self.obsYield),np.sum(self.obsGPP[:,120:273],axis=1),self.divideMonth(self.obsGPP[:,120:273])
    
    def _evaluate(self, x, out, *args, **kwargs):
        obs = self.decomposeObs()
        pre = self.simulation(x)
        self.loss1 = self.rmse(np.tile(obs[0],(self.enN,1)),np.stack(pre[0]))
        self.loss2 = self.rmse(np.tile(obs[1],(self.enN,1)),np.stack(pre[1]))/(5*30)
        self.loss3 = self.rmse(np.tile(obs[2],(self.enN,1)),np.stack(pre[2]))/(30)
        
        loss = self.loss1 + self.loss2 + self.loss3     
        out["F"] = loss

    def _calc_pareto_front(self):
        return 0.0

    def _calc_pareto_set(self):
        return np.full(self.n_var, 0)

def stackInput(data):
    inputSample = []
    for tmp in data:
        tmp = np.array(tmp)
        if tmp.shape[0]>365:
            t = tmp[:365,:]
        else:
            t=tmp
        inputSample.append(t)
    return np.stack(inputSample)

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
        cornFile = glob.glob('%s/*_corn.csv'%self.yieldPath)[0]
        soybeanFile = glob.glob('%s/*_soybean.csv'%self.yieldPath)[0]
        self.yield_NASS_corn = pd.read_csv(cornFile)
        self.yield_NASS_soybean = pd.read_csv(soybeanFile)

def lossTrajectory(res):
    n_evals = []             # corresponding number of function evaluations\
    hist_F = []              # the objective space values in each generation
    hist_cv = []             # constraint violation in each generation
    hist_cv_avg = []         # average constraint violation in the whole population
    
    for algo in res.history:
    
        # store the number of function evaluations
        n_evals.append(algo.evaluator.n_eval)
    
        # retrieve the optimum from the algorithm
        opt = algo.opt
    
        # store the least contraint violation and the average in each population
        hist_cv.append(opt.get("CV").min())
        hist_cv_avg.append(algo.pop.get("CV").mean())
    
        # filter out only the feasible and append and objective space values
        feas = np.where(opt.get("feasible"))[0]
        hist_F.append(opt.get("F")[feas])
    
def plotHistory(res):
    n_evals = np.array([e.evaluator.n_eval for e in res.history])
    opt = np.array([e.opt[0].F for e in res.history])
    f_ave = np.array([e.output.f_avg.value for e in res.history])
    loss1 = [np.mean(e.problem.loss1) for e in res.history]
    loss2 = [np.mean(e.problem.loss2) for e in res.history]
    loss3 = [np.mean(e.problem.loss3) for e in res.history]
    loss = np.array(loss1) + np.array(loss2) + np.array(loss3)
    fig = plt.figure()
    plt.title("Convergence")
    plt.plot(n_evals, f_ave, "k--", label='f ave')
    plt.plot(n_evals, opt, "b--", label='f min')
    plt.legend()
    
    fig = plt.figure()
    plt.title("Convergence")
    plt.plot(n_evals, f_ave, "k--", label='f ave')
    plt.plot(n_evals, opt, "b--", label='f min')
    plt.plot(n_evals, loss1, "y--",label='Yield loss')
    plt.plot(n_evals, loss2, "r--",label='GPPsum loss')
    plt.plot(n_evals, loss3, "g--",label='GPP month loss')
    # plt.plot(n_evals, loss, "k--",label='loss_all')
    plt.legend()
    # plt.yscale("log")
    return fig

def plotRes(pre_best,pre_origin,obs,coef):

    rmse = np.sqrt(np.nanmean((pre_best-obs)**2))
    fig= plt.figure(figsize=(16,9))
    ax = plt.subplot(1,1,1)
    ax.plot(pre_best/coef,color='black',linestyle='solid', label='Best objf.=%.4f'%rmse)
    ax.plot(pre_origin/coef,color='b',linestyle='solid', label='Orginal predict')
    ax.plot(obs/coef,'r.',markersize=5, label='Observation data')
    plt.xlabel('Number of Observation Points')
    plt.ylabel ('Discharge [l s-1]')
    plt.legend(loc='upper right')
    return fig

if __name__ == '__main__':
    model = ECONET() 
    
    # setting
    note='2000_2020_evenYear'
    n_gen=30
    yearRange_cali = [t for t in range(2000,2020+1,2)]
    # yearRange_cali = [t for t in range(2000,2020+1)]
    outPath = 'PSO_econet_gen%d_%s'%(n_gen,note)
    dataRoot = 'demoData/calirationData'  
    GPPpath = dataRoot
    saveHistory = False
    
    mkdir(outPath)
    
    dic_para_corn = {}
    dic_para_soybean = {}
    # crop = 'soy' #'corn'
    for crop in ['corn','soy']:
    # for crop in ['soy']:    
        # obs NASS yield
        obsType = [-1,2] # yield
        NASS_Path = dataRoot
        NASS_yield = yieldValidation(NASS_Path)
        if crop == 'corn':
            df_yield = NASS_yield.yield_NASS_corn
            cropType=0
        else:
            df_yield = NASS_yield.yield_NASS_soybean
            cropType=1
            
        # input data 
        FIPSList = ['17049']
           
        for n,FIPS in enumerate(FIPSList):
            print('Processing FIPS %s, %s...'%(FIPS,crop))
            if os.path.exists('%s/PSO_%s_%s.csv'%(outPath,FIPS,crop)):
                continue
               
            # load input and Ecosys output data
            inputDataPath = '%s/%s_inputMerged.pkl'%(dataRoot,FIPS)
            inputMerged = load_object(inputDataPath)
              
            genEn = makeInput([inputMerged])
            inputEpisode_reset = []
            obs = []
            obs2 = []
            yearRange_vali = []
            # fetch and process obs data
            if crop=='corn':
                tmp = load_object('%s/GPP_%s_%s.pkl'%(GPPpath,FIPS,crop))
            else:
                tmp = load_object('%s/GPP_%s_%sbean.pkl'%(GPPpath,FIPS,crop))
            for year in yearRange_cali:
                inputEpisodes = genEn.get(year)
                inputEpisode_reset.append(genEn.resetDefaultPara(inputEpisodes,cropTypes=[cropType])[0])
                obs.append(df_yield[df_yield['Year']==year][FIPS].item() / NASS_yield.coef_C2BUacre(cropType) * genEn.y_NormCoef[obsType[0]])
                obs2.append(tmp[year]* genEn.y_NormCoef[obsType[1]])
            
            # remove None in inputEpisodes
            inputEpisode_reset_vali = []
            obs_vali = []
            obs2_vali = []
            for t,k,z,y in zip(inputEpisode_reset,obs,obs2,yearRange_cali):
                if t is not None:
                    inputEpisode_reset_vali.append(t)
                    obs_vali.append(k)
                    obs2_vali.append(z)
                    yearRange_vali.append(y)
            inputTemplet = stackInput(inputEpisode_reset_vali).astype(np.float32)
            
            problem = myProblem(crop=crop, model=model, inputTemplet=inputTemplet,obsYield=obs_vali,obsGPP=obs2_vali,
                                                       yearRange=yearRange_vali,obsType=obsType)

            algorithm = PSO()
            termination = get_termination("n_gen", n_gen)
            res = minimize(problem,
                           algorithm,
                           termination,
                           seed=1,
                           verbose=True,
                           save_history=True)

            print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
            print("Calibration time: %.2f s"%res.exec_time)
            # plot results
            para_cali = res.X
            if n==0:
                # # plot rmse trend
                fig = plotHistory(res)

                # original run
                out_origin =model.run(inputTemplet)
                pre_o1,pre_o2,pre_o3 = problem.splitOut([out_origin])
                  
                # # plot best vs. original            
                inputCali = problem.replaceInpput([para_cali])[0]
                out_cali =model.run(inputCali)
                pre_cali1,pre_cali2,pre_cali3 = problem.splitOut([out_cali])
               
                obs1,obs2,obs3 = problem.decomposeObs()     
                # yield
               
                fig = plotRes(pre_best=pre_cali1[0],pre_origin=pre_o1[0],obs=obs1,coef=genEn.y_NormCoef[obsType[0]])
                fig.savefig('%s/example_yield_%s_%s.png'%(outPath,FIPS,crop))
              
                # # sum growing season GPP
                fig = plotRes(pre_best=pre_cali2[0],pre_origin=pre_o2[0],obs=obs2,coef=genEn.y_NormCoef[obsType[1]])
                fig.savefig('%s/example_GPPseason_%s_%s.png'%(outPath,FIPS,crop))
                
                # # monthly growing season GPP
                fig = plotRes(pre_best=pre_cali3[0],pre_origin=pre_o3[0],obs=obs3,coef=genEn.y_NormCoef[obsType[1]])
                fig.savefig('%s/example_GPPmonth_%s_%s.png'%(outPath,FIPS,crop))
            # results
            if saveHistory:
                save_object(res.history,'%s/history_%s_%s.pkl'%(outPath,FIPS,crop))
            if crop == 'corn':
                dic_para_corn[FIPS] = para_cali
            else:
                dic_para_soybean[FIPS] = para_cali

    df_para_corn = pd.DataFrame(dic_para_corn)
    df_para_soybean = pd.DataFrame(dic_para_soybean)
    
    df_para_corn.to_csv('%s/PSO_para_corn.csv'%(outPath))
    df_para_soybean.to_csv('%s/PSO_para_soybean.csv'%(outPath))    
