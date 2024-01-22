# -*- coding: utf-8 -*-
"""
Created on Sun May  8 21:25:34 2022

@author: yang8460

2022-8-29: merged 5000 dataset, 10 steps take 211s
"""
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import sys
import os
import pandas as pd
import glob
import time
from scipy import stats
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

class loadPara():
    def __init__(self,path,crop,algorithm='PSO',mode='inteval',version = ''):
        # calibrated parameters
        self.path = path
        if crop=='soy':
            crop='soybean'
        self.crop=crop
        self.para_cali = None
        self.algorithm=algorithm
        self.mode=mode
        self.version=version
        
        if self.crop == 'corn':
            self.paraLoc = [5,8,9,10,11,12,20]
        else:           
            self.paraLoc = [5,7,9,10,11,12]
        self.paraDic = {}
        
    def getPara(self,year):
        if (self.mode == 'intevalAcc')|(self.mode == 'inteval'):
            self.interval=3
            self.nodeList = [t for t in range(2000,2019,3)]
            if year< self.nodeList[1]:
                return None
            
            else:
                if year >= 2018:
                    end=2018
                else:
                    for i in range(len(self.nodeList)-1):
                        if (self.nodeList[i]<=year)&(self.nodeList[i+1]>year):
                            end = self.nodeList[i]
                            break
                
                if end in self.paraDic.keys():
                    return self.paraDic[end]
                else:
                    if self.mode == 'intevalAcc':
                        start = 2000
                        gen=30
                    elif self.mode == 'inteval':
                        start = end - self.interval
                        gen=30
                    self.para_cali = pd.read_csv('%s/PSO_econet_gen%s_interval%s_%s_%s%s/%s_para_%s.csv'%(self.path,gen,self.interval,start,
                                                                                                        end,self.version,self.algorithm,self.crop))
                    self.paraDic[end] = self.para_cali
                    return self.para_cali
        elif self.mode == 'eachYear':
            self.interval=1
            start=year
            end=year
            self.para_cali = pd.read_csv('%s/PSO_econet_gen30_interval%s_%s_%s%s/%s_para_%s.csv'%(self.path,self.interval,start,
                                                                                              end,self.version,self.algorithm,self.crop))
            return self.para_cali
        elif self.mode == 'previousYear':
            if year==2000:
                return None
            else:
                self.interval=1
                start=year-1
                end=year-1
                self.para_cali = pd.read_csv('%s/PSO_econet_gen30_interval%s_%s_%s%s/%s_para_%s.csv'%(self.path,self.interval,start,
                                                                                                    end,self.version,self.algorithm,self.crop))
                return self.para_cali
        elif self.mode == 'global':
            self.interval=3
            start=2000
            end=2018
            self.para_cali = pd.read_csv('%s/PSO_econet_gen30_interval%s_%s_%s%s/%s_para_%s.csv'%(self.path,self.interval,start,
                                                                                                end,self.version,self.algorithm,self.crop))            
            return self.para_cali
        
        elif self.mode == 'default':
            return None
        
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

class loadCDLdata():
    '''
    added threshold to trim samples, 8/30/2022
    '''
    def __init__(self,CDLfile,years = 21,threshold=0.8):
        self.years = years
        self.threshold= threshold
        self.sitesCDL = pd.read_csv(CDLfile)
        self.sitesCDL.drop(['system:index','2005b','2007b'],axis=1,inplace=True)
        self.sitesCDL.rename(columns = {'2005a' : '2005', '2007a' : '2007'}, inplace = True)
        tmp = np.array(self.sitesCDL)[:,:years]
        tmp[tmp=='No data'] = np.nan
        tmp = tmp.astype(np.float32)
        tmp2 = np.zeros((tmp.shape[0],tmp.shape[1]))
        tmp2[(tmp!=1)&(tmp!=5)] = 1
        pure_ratio = 1-np.sum(tmp2,axis=1)/years
        unpure_index = np.where(pure_ratio<=threshold)[0]
        self.sitesCDL.drop(index = unpure_index,inplace=True)
        
    def get(self,year,site):
        tmp = self.sitesCDL.loc[self.sitesCDL['Site']==site][str(year)]
        if len(tmp)==0:
            CDL = 'No data'
        else:
            CDL = tmp.item()
        if CDL == 'No data':
            return None
        elif float(CDL) == 1:
            return 0
        elif float(CDL) == 5:
            return 1
        else:
            return None
        
    def getCornSite(self,year):
        tmp = self.sitesCDL[str(year)].copy()
        tmp[tmp=='No data'] = np.nan
        cornSites = self.sitesCDL[tmp.astype(float) == 1]['Site']

        return cornSites
    
    def getSoybeanSite(self,year):
        tmp = self.sitesCDL[str(year)].copy()
        tmp[tmp=='No data'] = np.nan
        SoybeanSites = self.sitesCDL[tmp.astype(float) == 5]['Site']

        return SoybeanSites
    
    def getCornSiteFIPS(self,year,FIPS):
        tmp = self.sitesCDL[str(year)].copy()
        tmp[tmp=='No data'] = np.nan

        cornSites = self.sitesCDL[(tmp.astype(float) == 1)&(self.sitesCDL['FIPS']==FIPS)]['Site']
        return cornSites
    
    def getSoybeanSiteFIPS(self,year,FIPS):
        tmp = self.sitesCDL[str(year)].copy()
        tmp[tmp=='No data'] = np.nan

        SoybeanSites = self.sitesCDL[(tmp.astype(float) == 5)&(self.sitesCDL['FIPS']==FIPS)]['Site']
        return SoybeanSites
    
    def judge(self,site):
        siteIndex = self.sitesCDL['Site']==site
        if np.max(siteIndex):
            CDL = np.array(self.sitesCDL.loc[siteIndex].iloc[0][:self.years])
        else:
            return False
        CDL[np.where(CDL=='No data')] = np.nan
        CDL = CDL.astype(np.float32)
        loc = np.where((CDL==1) | (CDL==5))[0]
        if len(loc)/self.years < self.threshold:
            return False
        else:
            return True
        
def listSplit(l,ref_index):
    inList = []
    outList = []
    for i,t in enumerate(l):
        if i in ref_index:
            inList.append(t)
        else:
            outList.append(t)
    return inList,outList

def train_test_split_no_leak(X, y, test_ratio=0.1):
    np.random.seed(0)
    indexList =np.arange(0,len(X))
    np.random.shuffle(indexList)
    
    # split train and test with no leak strategy
    indexTest = indexList[:int(test_ratio*len(X))]
    X_test, X_train = listSplit(X,indexTest)
    y_test, y_train = listSplit(y,indexTest)
    
    return X_train, X_test, y_train, y_test
        
def train_val_test_split_no_leak(X, y, test_ratio=0.1):
    np.random.seed(0)
    val_ratio = test_ratio / (1 - test_ratio)
    indexList =np.arange(0,len(X))
    np.random.shuffle(indexList)
    
    # split train and test with no leak strategy
    indexTest = indexList[:int(test_ratio*len(X))]
    X_test, X_train_t = listSplit(X,indexTest)
    y_test, y_train_t = listSplit(y,indexTest)
    
    # split train and vali
    X_val = []
    X_train = []
    y_val = []
    y_train = []
    for x_s,y_s in zip(X_train_t,y_train_t):
        indexYear =np.arange(0,len(x_s))
        np.random.shuffle(indexYear)
        indexVali = indexYear[:int(val_ratio*len(x_s))]
        tmp_v,tmp_t = listSplit(x_s,indexVali)
        X_train.append(tmp_t)
        X_val.append(tmp_v)
        tmp_v,tmp_t = listSplit(y_s,indexVali)
        y_train.append(tmp_t)
        y_val.append(tmp_v)
   
    return X_train, X_val, X_test, y_train, y_val, y_test

class EcoNet_dataset(Dataset):
    def __init__(self, data_X,data_y, X_selectFeatures = None, y_selectFeatures = None,y_NormCoef=None):
        self.data_X = data_X
        self.data_y = data_y
        self.X_selectFeatures = X_selectFeatures
        self.y_selectFeatures = y_selectFeatures
        self.y_NormCoef = y_NormCoef
        
    def __getitem__(self, index, batchmodel = True):
        
        if batchmodel:
            index_s, index_y = index
            if self.X_selectFeatures == None:
                out_X = np.array(self.data_X[index_s][index_y]).astype(np.float32)
            else:
                out_X = np.array(self.data_X[index_s][index_y][self.X_selectFeatures]).astype(np.float32)
            if self.y_selectFeatures == None:           
                out_y = np.array(self.data_y[index_s][index_y]).astype(np.float32)
            else:
                out_y = np.array(self.data_y[index_s][index_y][self.y_selectFeatures]).astype(np.float32)
            
            if not self.y_NormCoef==None:
                for i,t in enumerate(self.y_NormCoef):
                    out_y[:,i] = out_y[:,i]*t
                               
        else:
            index_s, index_y, index_d = index
            if self.X_selectFeatures == None:
                out_X = self.data_X[index_s][index_y].iloc[index_d].tolist()
            else:
                out_X = self.data_X[index_s][index_y][self.X_selectFeatures].iloc[index_d].tolist()
            if self.y_selectFeatures == None:           
                out_y = self.data_y[index_s][index_y].iloc[index_d].tolist()
            else:
                out_y = self.data_y[index_s][index_y][self.y_selectFeatures].iloc[index_d].tolist()
            
            if not self.y_NormCoef==None:
                for i,t in enumerate(self.y_NormCoef):
                    out_y[i] = out_y[i]*t
                    
        return out_X, out_y
    
    def __len__(self):
        return len(self.data_X)

class EcoNet_dataset_site(Dataset):
    def __init__(self, data_X,data_y, X_selectFeatures = None, y_selectFeatures = None,y_NormCoef=None):
        self.data_X = data_X
        self.data_y = data_y
        self.X_selectFeatures = X_selectFeatures
        self.y_selectFeatures = y_selectFeatures
        self.y_NormCoef = y_NormCoef
        
    def __getitem__(self, index, batchmodel = True):
        
        if batchmodel:
            index_y = index
            if self.X_selectFeatures == None:
                out_X = np.array(self.data_X[index_y]).astype(np.float32)
            else:
                out_X = np.array(self.data_X[index_y][self.X_selectFeatures]).astype(np.float32)
            if self.y_selectFeatures == None:           
                out_y = np.array(self.data_y[index_y]).astype(np.float32)
            else:
                out_y = np.array(self.data_y[index_y][self.y_selectFeatures]).astype(np.float32)
            
            if not self.y_NormCoef==None:
                for i,t in enumerate(self.y_NormCoef):
                    out_y[:,i] = out_y[:,i]*t
                               
        else:
            index_y, index_d = index
            if self.X_selectFeatures == None:
                out_X = self.data_X[index_y].iloc[index_d].tolist()
            else:
                out_X = self.data_X[index_y][self.X_selectFeatures].iloc[index_d].tolist()
            if self.y_selectFeatures == None:           
                out_y = self.data_y[index_y].iloc[index_d].tolist()
            else:
                out_y = self.data_y[index_y][self.y_selectFeatures].iloc[index_d].tolist()
            
            if not self.y_NormCoef==None:
                for i,t in enumerate(self.y_NormCoef):
                    out_y[i] = out_y[i]*t
                    
        return out_X, out_y
    
    def __len__(self):
        return len(self.data_X)
    
class EcoNet_dataset_pkl(Dataset):
    def __init__(self, data_X_pathes, data_y_pathes, X_selectFeatures = None, y_selectFeatures = None,y_NormCoef=None):
        self.data_X_pathes = data_X_pathes
        self.data_y_pathes = data_y_pathes
        self.X_selectFeatures = X_selectFeatures
        self.y_selectFeatures = y_selectFeatures
        self.y_NormCoef = y_NormCoef
        
    def __getitem__(self, index, batchmodel = True):
        
        if batchmodel:
            index_s, index_y = index
            self.data_X = self.load_object(self.data_X_pathes[index_s])
            self.data_y = self.load_object(self.data_y_pathes[index_s])
            
            if self.X_selectFeatures == None:
                out_X = np.array(self.data_X[index_y]).astype(np.float32)
            else:
                out_X = np.array(self.data_X[index_y][self.X_selectFeatures]).astype(np.float32)
            if self.y_selectFeatures == None:           
                out_y = np.array(self.data_y[index_y]).astype(np.float32)
            else:
                out_y = np.array(self.data_y[index_y][self.y_selectFeatures]).astype(np.float32)
            
            if not self.y_NormCoef==None:
                for i,t in enumerate(self.y_NormCoef):
                    out_y[:,i] = out_y[:,i]*t
                               
        else:
            index_s, index_y, index_d = index
            self.data_X = self.load_object(self.data_X_pathes[index_s])
            self.data_y = self.load_object(self.data_y_pathes[index_s])
            
            if self.X_selectFeatures == None:
                out_X = self.data_X[index_y].iloc[index_d].tolist()
            else:
                out_X = self.data_X[index_y][self.X_selectFeatures].iloc[index_d].tolist()
            if self.y_selectFeatures == None:           
                out_y = self.data_y[index_y].iloc[index_d].tolist()
            else:
                out_y = self.data_y[index_y][self.y_selectFeatures].iloc[index_d].tolist()
            
            if not self.y_NormCoef==None:
                for i,t in enumerate(self.y_NormCoef):
                    out_y[i] = out_y[i]*t
                    
        return out_X, out_y
    
    def load_object(self, filename):
        with open(filename, 'rb') as inp:
            data = pickle.load(inp)
        return data
    
    def __len__(self):
        return len(self.data_X_pathes)

class EcoNet_dataset_pkl_yearly(Dataset):
    def __init__(self, data_X_pathes, data_y_pathes, X_selectFeatures = None, y_selectFeatures = None,y_NormCoef=None,length = 365):
        self.data_X_pathes = data_X_pathes
        self.data_y_pathes = data_y_pathes
        self.X_selectFeatures = X_selectFeatures
        self.y_selectFeatures = y_selectFeatures
        self.y_NormCoef = y_NormCoef
        self.length = length
        
    def __getitem__(self, index, batchmodel = True):
        
        if batchmodel:

            self.data_X = self.load_object(self.data_X_pathes[index])
            self.data_y = self.load_object(self.data_y_pathes[index])
            
            if self.X_selectFeatures == None:
                out_X = np.array(self.data_X).astype(np.float32)
            else:
                out_X = np.array(self.data_X[self.X_selectFeatures]).astype(np.float32)
            if self.y_selectFeatures == None:           
                out_y = np.array(self.data_y).astype(np.float32)
            else:
                out_y = np.array(self.data_y[self.y_selectFeatures]).astype(np.float32)
            
            if not self.y_NormCoef==None:
                for i,t in enumerate(self.y_NormCoef):
                    out_y[:,i] = out_y[:,i]*t
            
            length = out_X.shape[0]
            if length >= self.length:
                out_X_pad = out_X[:self.length,:]
                out_y_pad = out_y[:self.length,:]
            else:
                out_X_pad = np.zeros((self.length,out_X.shape[1])).astype(np.float32)
                out_X_pad[:out_X.shape[0],:] = out_X
                out_y_pad = np.zeros((self.length,out_y.shape[1])).astype(np.float32)
                out_y_pad[:out_y.shape[0],:] = out_y
            return out_X_pad, out_y_pad, length
        else:
            index_s, index_d = index
            self.data_X = self.load_object(self.data_X_pathes[index_s])
            self.data_y = self.load_object(self.data_y_pathes[index_s])
            
            if self.X_selectFeatures == None:
                out_X = self.data_X.iloc[index_d].tolist()
            else:
                out_X = self.data_X[self.X_selectFeatures].iloc[index_d].tolist()
            if self.y_selectFeatures == None:           
                out_y = self.data_y.iloc[index_d].tolist()
            else:
                out_y = self.data_y[self.y_selectFeatures].iloc[index_d].tolist()
            
            if not self.y_NormCoef==None:
                for i,t in enumerate(self.y_NormCoef):
                    out_y[i] = out_y[i]*t
                    
                    return out_X, out_y
    
    def load_object(self, filename):
        with open(filename, 'rb') as inp:
            data = pickle.load(inp)
        return data
    
    def __len__(self):
        return len(self.data_X_pathes)
    
class EcoNet_dataloader():
    def __init__(self,dataset,batch_size,sampleShuffle = True, discardLast = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampleShuffle = sampleShuffle
        self.discardLast = discardLast
        self.sampleIndexList = [list(np.arange(len(t))) for t in dataset.data_y]
        self.yearN = len(self.sampleIndexList[0])
        self.episodeN = len(self.sampleIndexList)*self.yearN
        self.epoch_batches = int(self.episodeN/batch_size)
        self.epochStart()
               
    def epochStart(self):
        self.remainYearList = self.sampleIndexList.copy()
        self.remainSampleList = list(np.arange(len(self.remainYearList)))
        self.currentSampleYear = [0,0]

    def getbatch(self):
        
        batchIndex = self.getbatchIndex()

        if len(batchIndex) > 0:
            X_batch=[]
            y_batch=[]
            for index in batchIndex:
                 
                X, y = self.dataset[(index[0],index[1])]
                    
                X_batch.append(torch.tensor(X))
                y_batch.append(torch.tensor(y))
            X_lengths = [sentence.shape[0] for sentence in X_batch]
            X_batch_pad = torch.nn.utils.rnn.pad_sequence(X_batch, batch_first=True)
            y_batch_pad = torch.nn.utils.rnn.pad_sequence(y_batch, batch_first=True)
            
            return X_batch_pad, y_batch_pad, X_lengths
        else:
            self.epochStart()
            return False
        
    def getbatchIndex(self):
        
        batchIndex = []
        for i in range(self.batch_size):   
            if len(self.remainSampleList) == 0:    
                if len(batchIndex) >0:
                    if self.discardLast:
                        # print('last batch discarded, length is %d'%(len(batchIndex)))
                        return []
                    else:
                        # print('last batch used, length is %d'%(len(batchIndex)))
                        return batchIndex
                else:
                    # print('epoch finished!')
                    return []
            # select the sample 
            if self.sampleShuffle:
                selectedSample = random.sample(self.remainSampleList,1)[0]              
            else:
                selectedSample = self.remainSampleList[0]
            self.currentSampleYear[0] = selectedSample
            
            # select the year
            if self.sampleShuffle:
                selectedYear = random.sample(self.remainYearList[selectedSample],1)[0]
            else:
                selectedYear = self.remainYearList[selectedSample][0]
            self.currentSampleYear[1] = selectedYear
                              
            # get batchIndex
            batchIndex.append(self.currentSampleYear.copy())
    
            # update sample list by removing the sampled index
            self.remainYearList[self.currentSampleYear[0]] = list(set(
                    self.remainYearList[self.currentSampleYear[0]]) ^ set([selectedYear]))
            self.remainYearList[self.currentSampleYear[0]].sort()
            
            # if the sample site have no more data after this batch
            if len(self.remainYearList[self.currentSampleYear[0]]) == 0:
                self.remainSampleList = list(set(self.remainSampleList) ^ set([selectedSample]))
                self.remainSampleList.sort()
         
        return batchIndex

class EcoNet_dataloader_site():
    def __init__(self,dataset,batch_size,sampleShuffle = True, discardLast = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampleShuffle = sampleShuffle
        self.discardLast = discardLast
        self.yearIndexList = list(np.arange(len(dataset.data_y)))

        self.epochStart()
               
    def epochStart(self):
        self.remainYearList = self.yearIndexList.copy()

        self.currentYear = 0

    def getbatch(self):
        
        batchIndex = self.getbatchIndex()

        if len(batchIndex) > 0:
            X_batch=[]
            y_batch=[]
            for index in batchIndex:
                 
                X, y = self.dataset[index]
                    
                X_batch.append(torch.tensor(X))
                y_batch.append(torch.tensor(y))
            X_lengths = [sentence.shape[0] for sentence in X_batch]
            X_batch_pad = torch.nn.utils.rnn.pad_sequence(X_batch, batch_first=True)
            y_batch_pad = torch.nn.utils.rnn.pad_sequence(y_batch, batch_first=True)
            
            return X_batch_pad, y_batch_pad, X_lengths
        else:
            self.epochStart()
            return False
        
    def getbatchIndex(self):
        
        batchIndex = []
        for i in range(self.batch_size):   
            if len(self.remainYearList) == 0:    
                if len(batchIndex) >0:
                    if self.discardLast:
                        # print('last batch discarded, length is %d'%(len(batchIndex)))
                        return []
                    else:
                        # print('last batch used, length is %d'%(len(batchIndex)))
                        return batchIndex
                else:
                    # print('epoch finished!')
                    return []
            
            # select the year
            if self.sampleShuffle:
                selectedYear = random.sample(self.remainYearList,1)[0]
            else:
                selectedYear = self.remainYearList[0]
            self.currentYear = selectedYear
                              
            # get batchIndex
            batchIndex.append(self.currentYear.copy())
    
            # update sample list by removing the sampled index
            self.remainYearList = list(set(
                    self.remainYearList) ^ set([selectedYear]))
            self.remainYearList.sort()
                     
        return batchIndex
    
class EcoNet_dataloader_pkl():
    def __init__(self,dataset,batch_size,sampleShuffle = True, discardLast = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampleShuffle = sampleShuffle
        self.discardLast = discardLast
        self.yearN = len(self.dataset.load_object(self.dataset.data_y_pathes[0]))
        self.sampleIndexList = [list(np.arange(self.yearN)) for t in range(len(self.dataset.data_y_pathes))]
        self.episodeN = len(self.sampleIndexList)*self.yearN
        self.epoch_batches = int(self.episodeN/batch_size)
        self.epochStart()
               
    def epochStart(self):
        self.remainYearList = self.sampleIndexList.copy()
        self.remainSampleList = list(np.arange(len(self.remainYearList)))
        self.currentSampleYear = [0,0]

    def getbatch(self):
        
        batchIndex = self.getbatchIndex()

        if len(batchIndex) > 0:
            X_batch=[]
            y_batch=[]
            for index in batchIndex:
                 
                X, y = self.dataset[(index[0],index[1])]
                    
                X_batch.append(torch.tensor(X))
                y_batch.append(torch.tensor(y))
            X_lengths = [sentence.shape[0] for sentence in X_batch]
            X_batch_pad = torch.nn.utils.rnn.pad_sequence(X_batch, batch_first=True)
            y_batch_pad = torch.nn.utils.rnn.pad_sequence(y_batch, batch_first=True)
            
            return X_batch_pad, y_batch_pad, X_lengths
        else:
            self.epochStart()
            return False
        
    def getbatchIndex(self):
        
        batchIndex = []
        for i in range(self.batch_size):   
            if len(self.remainSampleList) == 0:    
                if len(batchIndex) >0:
                    if self.discardLast:
                        # print('last batch discarded, length is %d'%(len(batchIndex)))
                        return []
                    else:
                        # print('last batch used, length is %d'%(len(batchIndex)))
                        return batchIndex
                else:
                    # print('epoch finished!')
                    return []
            # select the sample 
            if self.sampleShuffle:
                selectedSample = random.sample(self.remainSampleList,1)[0]              
            else:
                selectedSample = self.remainSampleList[0]
            self.currentSampleYear[0] = selectedSample
            
            # select the year
            if self.sampleShuffle:
                selectedYear = random.sample(self.remainYearList[selectedSample],1)[0]
            else:
                selectedYear = self.remainYearList[selectedSample][0]
            self.currentSampleYear[1] = selectedYear
                              
            # get batchIndex
            batchIndex.append(self.currentSampleYear.copy())
    
            # update sample list by removing the sampled index
            self.remainYearList[self.currentSampleYear[0]] = list(set(
                    self.remainYearList[self.currentSampleYear[0]]) ^ set([selectedYear]))
            self.remainYearList[self.currentSampleYear[0]].sort()
            
            # if the sample site have no more data after this batch
            if len(self.remainYearList[self.currentSampleYear[0]]) == 0:
                self.remainSampleList = list(set(self.remainSampleList) ^ set([selectedSample]))
                self.remainSampleList.sort()
         
        return batchIndex
    
class EcoNet_dataloader_singleStep():
    def __init__(self,dataset,batch_size,sampleShuffle = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampleShuffle = sampleShuffle
        self.sampleIndexList = [list(np.arange(len(t))) for t in dataset.data_y]
        self.epochStart()
               
    def epochStart(self):
        self.remainYearList = self.sampleIndexList.copy()
        self.remainSampleList = list(np.arange(len(self.remainYearList)))
        self.currentSampleYear = [0,0]
        self.newEpisode = True

    def getbatch(self):
        
        index = self.getbatchIndex()

        if len(index) > 0:
            X_batch=[]
            y_batch=[]
            for i in index:
                X, y = self.dataset.__getitem__((self.currentSampleYear[0],self.currentSampleYear[1],i),batchmodel=False)
                X_batch.append(X)
                y_batch.append(y)
            
            return torch.tensor(np.array(X_batch).astype(np.float32)), torch.tensor(np.array(y_batch).astype(np.float32))
        else:
            self.epochStart()
            return False
        
    def getbatchIndex(self):
        
        # select the sample
        if self.newEpisode:
            if len(self.remainSampleList) == 0:
                print('epoch finished!')
                return []
            if self.sampleShuffle:
                selectedSample = random.sample(self.remainSampleList,1)[0]              
            else:
                selectedSample = self.remainSampleList[0]
            self.currentSampleYear[0] = selectedSample
            
            # select the year
            if self.sampleShuffle:
                selectedYear = random.sample(self.remainYearList[selectedSample],1)[0]
            else:
                selectedYear = self.remainYearList[selectedSample][0]
            self.currentSampleYear[1] = selectedYear
            
        else:
            selectedSample = self.currentSampleYear[0]
            selectedYear = self.currentSampleYear[1]
            
        # growing season span
        if self.newEpisode:
            self.seasonSpan = list(np.arange(self.dataset.data_y[
                self.currentSampleYear[0]][self.currentSampleYear[1]].shape[0]))
        
        # get the batchIndex
        if len(self.seasonSpan) > self.batch_size:
            batchIndex = self.seasonSpan[:self.batch_size]
            self.newEpisode = False           
        else:
            batchIndex = self.seasonSpan.copy()
            self.newEpisode = True
            
        # update sample list by removing the sampled index
        self.seasonSpan = list(set(self.seasonSpan)^set(batchIndex))
        self.seasonSpan.sort()
        if self.newEpisode:
            
            self.remainYearList[self.currentSampleYear[0]] = list(set(
                    self.remainYearList[self.currentSampleYear[0]]) ^ set([selectedYear]))
            self.remainYearList[self.currentSampleYear[0]].sort()
            
            # if the sample site have no more data after this batch
            if len(self.remainYearList[self.currentSampleYear[0]]) == 0:
                self.remainSampleList = list(set(self.remainSampleList) ^ set([selectedSample]))
                self.remainSampleList.sort()
         
        return batchIndex   

class EcoNet_dataloader_singleStep_pkl():
    def __init__(self,dataset,batch_size,sampleShuffle = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampleShuffle = sampleShuffle
        self.yearN = len(self.dataset.load_object(self.dataset.data_y_pathes[0]))
        self.sampleIndexList = [list(np.arange(self.yearN)) for t in range(len(self.dataset.data_y_pathes))]
        self.episodeN = len(self.sampleIndexList)*self.yearN
        self.epoch_batches = int(self.episodeN/batch_size)
        self.epochStart()        
               
    def epochStart(self):
        self.remainYearList = self.sampleIndexList.copy()
        self.remainSampleList = list(np.arange(len(self.remainYearList)))
        self.currentSampleYear = [0,0]
        self.newEpisode = True

    def getbatch(self):
        
        index = self.getbatchIndex()

        if len(index) > 0:
            X_batch=[]
            y_batch=[]
            for i in index:
                X, y = self.dataset.__getitem__((self.currentSampleYear[0],self.currentSampleYear[1],i),batchmodel=False)
                X_batch.append(X)
                y_batch.append(y)
            
            return torch.tensor(np.array(X_batch).astype(np.float32)), torch.tensor(np.array(y_batch).astype(np.float32))
        else:
            self.epochStart()
            return False
        
    def getbatchIndex(self):
        
        # select the sample
        if self.newEpisode:
            if len(self.remainSampleList) == 0:
                print('epoch finished!')
                return []
            if self.sampleShuffle:
                selectedSample = random.sample(self.remainSampleList,1)[0]              
            else:
                selectedSample = self.remainSampleList[0]
            self.currentSampleYear[0] = selectedSample
            
            # select the year
            if self.sampleShuffle:
                selectedYear = random.sample(self.remainYearList[selectedSample],1)[0]
            else:
                selectedYear = self.remainYearList[selectedSample][0]
            self.currentSampleYear[1] = selectedYear
            
        else:
            selectedSample = self.currentSampleYear[0]
            selectedYear = self.currentSampleYear[1]
            
        # growing season span
        if self.newEpisode:
            self.seasonSpan = list(np.arange(self.dataset.data_y[
                self.currentSampleYear[0]][self.currentSampleYear[1]].shape[0]))
        
        # get the batchIndex
        if len(self.seasonSpan) > self.batch_size:
            batchIndex = self.seasonSpan[:self.batch_size]
            self.newEpisode = False           
        else:
            batchIndex = self.seasonSpan.copy()
            self.newEpisode = True
            
        # update sample list by removing the sampled index
        self.seasonSpan = list(set(self.seasonSpan)^set(batchIndex))
        self.seasonSpan.sort()
        if self.newEpisode:
            
            self.remainYearList[self.currentSampleYear[0]] = list(set(
                    self.remainYearList[self.currentSampleYear[0]]) ^ set([selectedYear]))
            self.remainYearList[self.currentSampleYear[0]].sort()
            
            # if the sample site have no more data after this batch
            if len(self.remainYearList[self.currentSampleYear[0]]) == 0:
                self.remainSampleList = list(set(self.remainSampleList) ^ set([selectedSample]))
                self.remainSampleList.sort()
         
        return batchIndex   
    
class Optimization_decoder:
    def __init__(self, model, loss_fn=None, optimizer=None, exp_lr_scheduler=None, hiera=True):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.exp_lr_scheduler = exp_lr_scheduler
        self.train_losses = []
        self.val_losses = []
        self.train_losses_main = []
        self.val_losses_main = []
        self.train_losses_de = []
        self.val_losses_de = []
        self.hiera = hiera
    
    def changePadValue(self,y,seq_length):
        # replace the padding to -999
        t = torch.nn.utils.rnn.pack_padded_sequence(y, seq_length,
                                                    batch_first=True, enforce_sorted=False)
        y_repad, _ = torch.nn.utils.rnn.pad_packed_sequence(t, batch_first=True,padding_value=-999)
        return y_repad.detach()
        
    def train_step(self, x, y, x_length):
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        if self.hiera:
            yhat, _, hTensor,hTensor_v = self.model(x, seq_lengthList=x_length,isTrain=True)
        else:
            yhat,hTensor,hTensor_v = self.model(x, seq_lengthList=x_length,isTrain=True)
        
        # replace the padding to -999     
        y_repad = self.changePadValue(y=y,seq_length=x_length)        
        
        # calculate the mask
        # loss_main = 0
        # for i in range(y.shape[-1]):
        #     y_repad_i = y_repad[:,:,i].detach()
        #     mask = (y_repad_i != -999).float()
            
        #     # Computes MSEloss
        #     yhat_i = yhat[:,:,i]
        #     y_i = y[:,:,i]
        #     loss_main += torch.sum(((yhat_i-y_i)*mask)**2) / torch.sum(mask)
        
        mask = (y_repad.detach() != -999).float()
        loss_main = torch.sum(((yhat-y)*mask)**2) / torch.sum(mask[:,:,0])
        
        # calculate reconstruction loss
        loss_de = 0
        for h,h_v in zip(hTensor,hTensor_v):
            if self.hiera:
                h_repad = self.changePadValue(y=h,seq_length=x_length)            
                mask = (h_repad != -999).float()
                
                loss_de += torch.sum(((h_v-h)*mask)**2) / torch.sum(mask)
            else:
                loss_de += torch.sum(((h_v-h))**2) / torch.sum(h)
            
        loss = loss_main + loss_de
        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item(), loss_main.item(), loss_de.item()

    def vali_step(self, x_val, y_val, x_val_length):
        self.model.eval()
        yhat, _, hTensor,hTensor_v = self.model(x_val,seq_lengthList=x_val_length,isTrain=True)
        
        # replace the padding to -999
        y_repad = self.changePadValue(y=y_val,seq_length=x_val_length)
        
        # calculate the mask
        val_loss_main = 0
        for i in range(y_val.shape[-1]):
            y_repad_i = y_repad[:,:,i].detach()
            mask = (y_repad_i != -999).float()
            
            # Computes MSEloss
            yhat_i = yhat[:,:,i]
            y_val_i = y_val[:,:,i]
            val_loss_main += torch.sum(((yhat_i-y_val_i)*mask)**2) / torch.sum(mask)    
       
        # calculate reconstruction loss
        val_loss_de = 0
        for h,h_v in zip(hTensor,hTensor_v):
            if self.hiera:
                h_repad = self.changePadValue(y=h,seq_length=x_val_length)            
                mask = (h_repad != -999).float()
                
                val_loss_de += torch.sum(((h_v-h)*mask)**2) / torch.sum(mask)
            else:
                val_loss_de += torch.sum(((h_v-h))**2) / torch.sum(h)
            
        val_loss = val_loss_main + val_loss_de
        return val_loss.item(), val_loss_main.item(), val_loss_de.item()
    
    def train(self, train_loader, val_loader, n_epochs=50, n_features=1, interv = 50, timeCount=False):

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            batch_losses_main = []
            batch_losses_de = []
            
            train_loader.epochStart()
            bN = 0
            print(f"[{epoch}/{n_epochs}] Training...")
            if timeCount:
                t_s = time.time()
                tt = time.time()
            while True:
                
                batch = train_loader.getbatch()
                if batch==False:
                    break
                x_batch, y_batch, x_length = batch
                if timeCount:
                    t_d = time.time()
                    dif_t =  t_d-tt
                    tt = t_d
                    print('load batch taking %.3f s'%dif_t)
                batch_size = x_batch.shape[0]
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)
                loss, loss_main, loss_de = self.train_step(x_batch, y_batch, x_length)
                batch_losses.append(loss)
                batch_losses_main.append(loss_main)
                batch_losses_de.append(loss_de)
                if timeCount:
                    t_d = time.time()
                    dif_t =  t_d-tt
                    tt = t_d
                    print('train taking %.3f s'%dif_t)
                bN+=1
                if bN % interv == 0:
                    if timeCount:
                        t_d = time.time()
                        dif_t =  t_d-t_s
                        t_s = t_d
                        print('Taking %.3f s'%dif_t)
                    print(f"[{epoch}/{n_epochs}] finished [{bN}/{train_loader.epoch_batches}] batches, ")
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)
            training_loss_main = np.mean(batch_losses_main)
            self.train_losses_main.append(training_loss_main)
            training_loss_de = np.mean(batch_losses_de)
            self.train_losses_de.append(training_loss_de)
            
            # lr decay
            self.exp_lr_scheduler.step()
            
            # validation
            print(f"[{epoch}/{n_epochs}] Validating...")
            
            with torch.no_grad():
                batch_val_losses = []
                batch_val_losses_main = []
                batch_val_losses_de = []
                val_loader.epochStart()
                while True:                   
                    batch_v = val_loader.getbatch()
                    if batch_v==False:
                        break
                    x_val, y_val, x_val_length = batch_v
                    batch_size_v = x_val.shape[0]
                    x_val = x_val.view([batch_size_v, -1, n_features]).to(device)
                    y_val = y_val.to(device)
                    
                    val_loss, val_loss_main, val_loss_de = self.vali_step(x_val, y_val, x_val_length)
                    batch_val_losses.append(val_loss)
                    batch_val_losses_main.append(val_loss_main)
                    batch_val_losses_de.append(val_loss_de)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)
                validation_loss_main = np.mean(batch_val_losses_main)
                self.val_losses_main.append(validation_loss_main)
                validation_loss_de = np.mean(batch_val_losses_de)
                self.val_losses_de.append(validation_loss_de)


            print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f} lr: {self.exp_lr_scheduler.get_last_lr()[0]:.6f}"
                )
            print(
                    f"[{epoch}/{n_epochs}] Training loss_main/decoder: [{training_loss_main:.4f}/{training_loss_de:.4f}]\t Validation loss_main/decoder: [{validation_loss_main:.4f}/{validation_loss_de:.4f}] "
                )
    def train_yearly(self, train_loader, val_loader, n_epochs=50, n_features=1, interv = 50, timeCount=False):

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            batch_losses_main = []
            batch_losses_de = []
            
            bN = 0
            print(f"[{epoch}/{n_epochs}] Training...")
            if timeCount:
                t_s = time.time()
                tt = time.time()
            for batch in train_loader:  
                x_batch, y_batch, x_length = batch
                # if timeCount:
                #     t_d = time.time()
                #     dif_t =  t_d-tt
                #     tt = t_d
                #     print('load batch taking %.3f s'%dif_t)
                batch_size = x_batch.shape[0]
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)
                loss, loss_main, loss_de = self.train_step(x_batch, y_batch, x_length)
                # if timeCount:
                #     t_d = time.time()
                #     dif_t =  t_d-tt
                #     tt = t_d
                #     print('train taking %.3f s'%dif_t)
                batch_losses.append(loss)
                batch_losses_main.append(loss_main)
                batch_losses_de.append(loss_de)
                bN+=1

                if bN % interv == 0:
                    if timeCount:
                        t_d = time.time()
                        dif_t =  t_d-t_s
                        t_s = t_d
                        print('Taking %.3f s'%dif_t)
                    print(f"[{epoch}/{n_epochs}] finished [{bN}/{len(train_loader)}] batches, ")
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)
            training_loss_main = np.mean(batch_losses_main)
            self.train_losses_main.append(training_loss_main)
            training_loss_de = np.mean(batch_losses_de)
            self.train_losses_de.append(training_loss_de)
            
            # lr decay
            self.exp_lr_scheduler.step()
            
            # validation
            print(f"[{epoch}/{n_epochs}] Validating...")
            
            with torch.no_grad():
                batch_val_losses = []
                batch_val_losses_main = []
                batch_val_losses_de = []

                for batch_v in val_loader:
                    x_val, y_val, x_val_length = batch_v
                    batch_size_v = x_val.shape[0]
                    x_val = x_val.view([batch_size_v, -1, n_features]).to(device)
                    y_val = y_val.to(device)
                    
                    val_loss, val_loss_main, val_loss_de = self.vali_step(x_val, y_val, x_val_length)
                    batch_val_losses.append(val_loss)
                    batch_val_losses_main.append(val_loss_main)
                    batch_val_losses_de.append(val_loss_de)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)
                validation_loss_main = np.mean(batch_val_losses_main)
                self.val_losses_main.append(validation_loss_main)
                validation_loss_de = np.mean(batch_val_losses_de)
                self.val_losses_de.append(validation_loss_de)


            print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f} lr: {self.exp_lr_scheduler.get_last_lr()[0]:.6f}"
                )
            print(
                    f"[{epoch}/{n_epochs}] Training loss_main/decoder: [{training_loss_main:.4f}/{training_loss_de:.4f}]\t Validation loss_main/decoder: [{validation_loss_main:.4f}/{validation_loss_de:.4f}] "
                )    
    def evaluate(self, test_loader, batch_size=1, n_features=1):
        with torch.no_grad():
            predictions = []
            values = []
            test_loader.epochStart()
            while True:
                batch = test_loader.getbatch()
                if batch==False:
                    break
                x_test, y_test, x_length = batch
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat,_ = self.model(x_test)
                predictions.append(yhat.cpu().detach().numpy())
                values.append(y_test.cpu().detach().numpy())

        return predictions, values

    def evaluate_yearly(self, test_loader, batch_size=1, n_features=1):
        with torch.no_grad():
            predictions = []
            values = []
  
            for batch in test_loader:
                x_test, y_test, x_length = batch
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat,_ = self.model(x_test)
                predictions.append(yhat.cpu().detach().numpy())
                values.append(y_test.cpu().detach().numpy())

        return predictions, values
    
    def evaluate_singleStep(self, test_loader, batch_size=1, n_features=1):
        with torch.no_grad():
            predictions = []
            values = []
            test_loader.epochStart()
            hidden_state = None
            while True:
                if test_loader.newEpisode == True:
                    hidden_state = None
                batch = test_loader.getbatch()
                if batch==False:
                    break
                x_test, y_test = batch
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat,hidden_state = self.model(x_test,hidden_state)
                predictions.append(yhat.cpu().detach().numpy())
                values.append(y_test.cpu().detach().numpy())    
        return predictions, values
    
    def plot_losses(self,outFolder=None,saveFig=False):
        fig=plt.figure()
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.xlabel('epoch')
        plt.ylabel('loss') 
        if saveFig:  
            fig.savefig('%s/loss.png'%outFolder)

class Optimization_decoder_VAE:
    def __init__(self, model, optimizer=None, exp_lr_scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.exp_lr_scheduler = exp_lr_scheduler
        self.train_losses = []
        self.val_losses = []
        self.train_losses_main = []
        self.val_losses_main = []
        self.train_losses_de = []
        self.val_losses_de = []
        
        self.train_losses_all = []
        self.val_losses_all = []
        self.train_losses_main_all = []
        self.val_losses_main_all = []
        self.train_losses_de_all = []
        self.val_losses_de_all = []
        
    def changePadValue(self,y,seq_length):
        # replace the padding to -999
        t = torch.nn.utils.rnn.pack_padded_sequence(y, seq_length,
                                                    batch_first=True, enforce_sorted=False)
        y_repad, _ = torch.nn.utils.rnn.pad_packed_sequence(t, batch_first=True,padding_value=-999)
        return y_repad.detach()
        
    def train_step(self, x, y, x_length,timeCount=False,showlog=False):
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        if timeCount:
            t_s = time.time()
            tt = time.time()
        yhat, _, hTensor,hTensor_v = self.model(x, seq_lengthList=x_length,isTrain=True)
        yhat_mu = yhat[0]
        yhat_sigma = yhat[1]
        if timeCount:
            t_d = time.time()
            dif_t =  t_d-tt
            tt = t_d
            print('inside train: net inference taking %.3f s'%dif_t)
        # replace the padding to -999     
        y_repad = self.changePadValue(y=y,seq_length=x_length)        
        
        # calculate the mask            
        mask = (y_repad.detach() != -999).float()
        # loss_main = torch.sum(((yhat_mu-y)*mask)**2) / torch.sum(mask[:,:,0])
        # negative_log_likelihood
        # loss_main = torch.mean(((y-yhat_mu)**2/(2*yhat_sigma**2) + torch.log(yhat_sigma**2)/2)*mask)
        loss_main = torch.sum(((y-yhat_mu)**2/(2*yhat_sigma**2) + 0.5*torch.log(yhat_sigma**2) + 0.5*np.log(2 * np.pi))*mask)/ torch.sum(mask[:,:,0])
        
        if showlog:        
            print('y {}, yhat_mu {}, yhat_sigma {}, nll_loss {}'.format(torch.mean(y).item(),
                                                                    torch.mean(yhat_mu).item(),torch.mean(yhat_sigma).item(),loss_main.item()))
        # calculate reconstruction loss
        loss_de = 0
        for h,h_v in zip(hTensor,hTensor_v):

            h_repad = self.changePadValue(y=h,seq_length=x_length)            
            mask = (h_repad != -999).float()
            
            loss_de += torch.sum(((h_v-h)*mask)**2) / torch.sum(mask)
            
        loss = loss_main + loss_de
        
        if timeCount:
            t_d = time.time()
            dif_t =  t_d-tt
            tt = t_d
            print('inside train: cal losses taking %.3f s'%dif_t)
        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        if timeCount:
            t_d = time.time()
            dif_t =  t_d-tt
            tt = t_d
            print('inside train: backprop taking %.3f s'%dif_t)
        # Returns the loss
        return loss.item(), loss_main.item(), loss_de.item()

    def vali_step(self, x_val, y_val, x_val_length):
        self.model.eval()
        yhat, _, hTensor,hTensor_v = self.model(x_val,seq_lengthList=x_val_length,isTrain=True)
        yhat_mu = yhat[0]
        yhat_sigma = yhat[1]
        
        # replace the padding to -999
        y_repad = self.changePadValue(y=y_val,seq_length=x_val_length)
        
        # calculate the mask            
        mask = (y_repad.detach() != -999).float()
            
        # Computes NLLloss
        val_loss_main = torch.sum(((y_val-yhat_mu)**2/(2*yhat_sigma**2) + 0.5*torch.log(yhat_sigma**2) + 0.5*np.log(2 * np.pi))*mask)/ torch.sum(mask[:,:,0])
       
        # calculate reconstruction loss
        val_loss_de = 0
        for h,h_v in zip(hTensor,hTensor_v):

            h_repad = self.changePadValue(y=h,seq_length=x_val_length)            
            mask = (h_repad != -999).float()
            
            val_loss_de += torch.sum(((h_v-h)*mask)**2) / torch.sum(mask)
                       
        val_loss = val_loss_main + val_loss_de
        return val_loss.item(), val_loss_main.item(), val_loss_de.item()
    
    def train(self, train_loader, val_loader, n_epochs=50, n_features=1, interv = 50, timeCount=False):
        self.interv = interv
        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            batch_losses_main = []
            batch_losses_de = []
            
            train_loader.epochStart()
            bN = 0
            print(f"[{epoch}/{n_epochs}] Training...")
            if timeCount:
                t_s = time.time()
                tt = time.time()
            while True:
                
                batch = train_loader.getbatch()
                if batch==False:
                    break
                x_batch, y_batch, x_length = batch
                if timeCount:
                    t_d = time.time()
                    dif_t =  t_d-tt
                    tt = t_d
                    print('load batch taking %.3f s'%dif_t)
                batch_size = x_batch.shape[0]
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)
                
                bN+=1
                if bN % self.interv== 0:
                    showlog = True
                else:
                    showlog = False
                loss, loss_main, loss_de = self.train_step(x_batch, y_batch, x_length,showlog=showlog)
                batch_losses.append(loss)
                batch_losses_main.append(loss_main)
                batch_losses_de.append(loss_de)
                if timeCount:
                    t_d = time.time()
                    dif_t =  t_d-tt
                    tt = t_d
                    print('train taking %.3f s'%dif_t)
                
                if bN % self.interv== 0:
                    if timeCount:
                        t_d = time.time()
                        dif_t =  t_d-t_s
                        t_s = t_d
                        print('Taking %.3f s'%dif_t)
                    print(f"[{epoch}/{n_epochs}] finished [{bN}/{train_loader.epoch_batches}] batches, ")
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)
            self.train_losses_all.extend(batch_losses)
            
            training_loss_main = np.mean(batch_losses_main)
            self.train_losses_main.append(training_loss_main)
            self.train_losses_main_all.extend(batch_losses_main)
            
            training_loss_de = np.mean(batch_losses_de)
            self.train_losses_de.append(training_loss_de)
            self.train_losses_de_all.extend(batch_losses_de)
            
            # lr decay
            self.exp_lr_scheduler.step()
            
            # validation
            print(f"[{epoch}/{n_epochs}] Validating...")
            
            with torch.no_grad():
                batch_val_losses = []
                batch_val_losses_main = []
                batch_val_losses_de = []
                val_loader.epochStart()
                while True:                   
                    batch_v = val_loader.getbatch()
                    if batch_v==False:
                        break
                    x_val, y_val, x_val_length = batch_v
                    batch_size_v = x_val.shape[0]
                    x_val = x_val.view([batch_size_v, -1, n_features]).to(device)
                    y_val = y_val.to(device)
                    
                    val_loss, val_loss_main, val_loss_de = self.vali_step(x_val, y_val, x_val_length)
                    batch_val_losses.append(val_loss)
                    batch_val_losses_main.append(val_loss_main)
                    batch_val_losses_de.append(val_loss_de)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)
                self.val_losses_all.extend(batch_val_losses)
                
                validation_loss_main = np.mean(batch_val_losses_main)
                self.val_losses_main.append(validation_loss_main)
                self.val_losses_main_all.extend(batch_val_losses_main)
                
                validation_loss_de = np.mean(batch_val_losses_de)
                self.val_losses_de.append(validation_loss_de)
                self.val_losses_de_all.extend(batch_val_losses_de)

            print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f} lr: {self.exp_lr_scheduler.get_last_lr()[0]:.6f}"
                )
            print(
                    f"[{epoch}/{n_epochs}] Training loss_main/decoder: [{training_loss_main:.4f}/{training_loss_de:.4f}]\t Validation loss_main/decoder: [{validation_loss_main:.4f}/{validation_loss_de:.4f}] "
                )
    def train_yearly(self, train_loader, val_loader, n_epochs=50, n_features=1, interv = 50, timeCount=False):

        for epoch in range(1, n_epochs + 1):
 
            batch_losses = []
            batch_losses_main = []
            batch_losses_de = []
            
            bN = 0
            print(f"[{epoch}/{n_epochs}] Training...")
            if timeCount:
                t_s = time.time()
                tt = time.time()
            for batch in train_loader:  
                x_batch, y_batch, x_length = batch
                if timeCount:
                    t_d = time.time()
                    dif_t =  t_d-tt
                    tt = t_d
                    print('load batch taking %.3f s'%dif_t)
                batch_size = x_batch.shape[0]
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)
                bN+=1
                if bN % interv== 0:
                    showlog = True
                else:
                    showlog = False
                
                loss, loss_main, loss_de = self.train_step(x_batch, y_batch, x_length, timeCount,showlog=showlog)
                if timeCount:
                    t_d = time.time()
                    dif_t =  t_d-tt
                    tt = t_d
                    print('train taking %.3f s'%dif_t)
               
                batch_losses.append(loss)
                batch_losses_main.append(loss_main)
                batch_losses_de.append(loss_de)

                if bN % interv == 0:
                    print('loss {}, loss_main {}, loss_de {}'.format(loss, loss_main, loss_de))
                    if timeCount:
                        t_d = time.time()
                        dif_t =  t_d-t_s
                        t_s = t_d
                        print('Taking %.3f s'%dif_t)
                    print(f"[{epoch}/{n_epochs}] finished [{bN}/{len(train_loader)}] batches, ")
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)
            self.train_losses_all.extend(batch_losses)
            
            training_loss_main = np.mean(batch_losses_main)
            self.train_losses_main.append(training_loss_main)
            self.train_losses_main_all.extend(batch_losses_main)
            
            training_loss_de = np.mean(batch_losses_de)
            self.train_losses_de.append(training_loss_de)
            self.train_losses_de_all.extend(batch_losses_de)
            
            # lr decay
            self.exp_lr_scheduler.step()
            
            # validation
            print(f"[{epoch}/{n_epochs}] Validating...")
            
            with torch.no_grad():
                batch_val_losses = []
                batch_val_losses_main = []
                batch_val_losses_de = []

                for batch_v in val_loader:
                    x_val, y_val, x_val_length = batch_v
                    batch_size_v = x_val.shape[0]
                    x_val = x_val.view([batch_size_v, -1, n_features]).to(device)
                    y_val = y_val.to(device)
                    
                    val_loss, val_loss_main, val_loss_de = self.vali_step(x_val, y_val, x_val_length)
                    batch_val_losses.append(val_loss)
                    batch_val_losses_main.append(val_loss_main)
                    batch_val_losses_de.append(val_loss_de)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)
                self.val_losses_all.extend(batch_val_losses)
                
                validation_loss_main = np.mean(batch_val_losses_main)
                self.val_losses_main.append(validation_loss_main)
                self.val_losses_main_all.extend(batch_val_losses_main)
                
                validation_loss_de = np.mean(batch_val_losses_de)
                self.val_losses_de.append(validation_loss_de)
                self.val_losses_de_all.extend(batch_val_losses_de)

            print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f} lr: {self.exp_lr_scheduler.get_last_lr()[0]:.6f}"
                )
            print(
                    f"[{epoch}/{n_epochs}] Training loss_main/decoder: [{training_loss_main:.4f}/{training_loss_de:.4f}]\t Validation loss_main/decoder: [{validation_loss_main:.4f}/{validation_loss_de:.4f}] "
                )    
    def evaluate(self, test_loader, batch_size=1, n_features=1):
        with torch.no_grad():
            predictions = []
            values = []
            test_loader.epochStart()
            while True:
                batch = test_loader.getbatch()
                if batch==False:
                    break
                x_test, y_test, x_length = batch
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat,_ = self.model(x_test)
                predictions.append(yhat.cpu().detach().numpy())
                values.append(y_test.cpu().detach().numpy())

        return predictions, values

    def evaluate_yearly(self, test_loader, batch_size=1, n_features=1):
        with torch.no_grad():
            predictions = []
            values = []
  
            for batch in test_loader:
                x_test, y_test, x_length = batch
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat,_ = self.model(x_test)
                yhat_mu = yhat[0]
                # yhat_sigma = yhat[1]
                predictions.append(yhat_mu.cpu().detach().numpy())
                values.append(y_test.cpu().detach().numpy())

        return predictions, values
    
    def evaluate_singleStep(self, test_loader, batch_size=1, n_features=1):
        with torch.no_grad():
            predictions = []
            values = []
            test_loader.epochStart()
            hidden_state = None
            while True:
                if test_loader.newEpisode == True:
                    hidden_state = None
                batch = test_loader.getbatch()
                if batch==False:
                    break
                x_test, y_test = batch
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat,hidden_state = self.model(x_test,hidden_state)
                predictions.append(yhat.cpu().detach().numpy())
                values.append(y_test.cpu().detach().numpy())    
        return predictions, values
    
    def plot_losses(self,outFolder=None,saveFig=False):
        fig=plt.figure()
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.xlabel('epoch')
        plt.ylabel('loss') 
        if saveFig:  
            fig.savefig('%s/loss.png'%outFolder)
    
    def plot_losses_step(self,outFolder=None,saveFig=False):
        fig=plt.figure()
        plt.plot(self.train_losses_main_all, label="Training loss-main")
        plt.plot(self.val_losses_main_all, label="Validation loss-main")
        plt.legend()
        plt.title("Losses")
        plt.xlabel('step')
        plt.ylabel('loss') 
        if saveFig:  
            fig.savefig('%s/loss_main_step.png'%outFolder)
            
class Optimization_stateAsInput:
    def __init__(self, model, loss_fn=None, optimizer=None,exp_lr_scheduler=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
        self.exp_lr_scheduler = exp_lr_scheduler
    
    def changePadValue(self,y,seq_length):
        # replace the padding to -999
        t = torch.nn.utils.rnn.pack_padded_sequence(y, seq_length,
                                                    batch_first=True, enforce_sorted=False)
        y_repad, _ = torch.nn.utils.rnn.pad_packed_sequence(t, batch_first=True,padding_value=-999)
        return y_repad.detach()
        
    def train_step(self, x, y, x_length):
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        yhat, _ = self.model(x, seq_lengthList=x_length,isTrain=True)
        
        # replace the padding to -999     
        y_repad = self.changePadValue(y=y,seq_length=x_length)        
        
        # calculate the mask
        loss = 0
        for i in range(y.shape[-1]):
            y_repad_i = y_repad[:,:,i].detach()
            mask = (y_repad_i != -999).float()
            
            # Computes MSEloss
            yhat_i = yhat[:,:,i]
            y_i = y[:,:,i]
            loss += torch.sum(((yhat_i-y_i)*mask)**2) / torch.sum(mask)
     
        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()

    def train(self, train_loader, val_loader, n_epochs=50, n_features=1):

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            
            train_loader.epochStart()
            while True:

                batch = train_loader.getbatch()
                if batch==False:
                    break
                x_batch, y_batch, x_length = batch
                batch_size = x_batch.shape[0]
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)
                loss = self.train_step(x_batch, y_batch, x_length)
                batch_losses.append(loss)

            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)
            
            # lr decay
            self.exp_lr_scheduler.step()
            
            with torch.no_grad():
                batch_val_losses = []
                val_loader.epochStart()
                while True:                   
                    batch_v = val_loader.getbatch()
                    if batch_v==False:
                        break
                    x_val, y_val, x_val_length = batch_v
                    batch_size_v = x_val.shape[0]
                    x_val = x_val.view([batch_size_v, -1, n_features]).to(device)
                    y_val = y_val.to(device)
                    self.model.eval()
                    yhat,_ = self.model(x_val,seq_lengthList=x_val_length)
                    
                    # replace the padding to -999
                    y_repad = self.changePadValue(y=y_val,seq_length=x_val_length)
                    
                    # calculate the mask
                    val_loss = 0
                    for i in range(y_val.shape[-1]):
                        y_repad_i = y_repad[:,:,i].detach()
                        mask = (y_repad_i != -999).float()
                        
                        # Computes MSEloss
                        yhat_i = yhat[:,:,i]
                        y_val_i = y_val[:,:,i]
                        val_loss += torch.sum(((yhat_i-y_val_i)*mask)**2) / torch.sum(mask)
                    val_loss = val_loss.item()
                    batch_val_losses.append(val_loss)
                    
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f} lr: {self.exp_lr_scheduler.get_last_lr()[0]:.6f}"
                )

    def evaluate(self, test_loader, batch_size=1, n_features=1):
        with torch.no_grad():
            predictions = []
            values = []
            test_loader.epochStart()
            while True:
                batch = test_loader.getbatch()
                if batch==False:
                    break
                x_test, y_test, x_length = batch
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat,_ = self.model(x_test)
                predictions.append(yhat.cpu().detach().numpy())
                values.append(y_test.cpu().detach().numpy())

        return predictions, values

    def evaluate_singleStep(self, test_loader, batch_size=1, n_features=1):
        with torch.no_grad():
            predictions = []
            values = []
            test_loader.epochStart()
            hidden_state = None
            while True:
                if test_loader.newEpisode == True:
                    hidden_state = None
                batch = test_loader.getbatch()
                if batch==False:
                    break
                x_test, y_test = batch
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat,hidden_state = self.model(x_test,hidden_state)
                predictions.append(yhat.cpu().detach().numpy())
                values.append(y_test.cpu().detach().numpy())    
        return predictions, values
    
    def plot_losses(self,outFolder=None,saveFig=False):
        fig=plt.figure()
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.xlabel('epoch')
        plt.ylabel('loss') 
        if saveFig:  
            fig.savefig('%s/loss.png'%outFolder)
            
def evaluate_singleStep_episode(model, test_loader, batch_size=1, n_features=1, episodeN = None):
    with torch.no_grad():
        predictions = []
        values = []
        inputFeature = []
        test_loader.epochStart()
        hidden_state = None
        if episodeN == None:
            while True:
                if test_loader.newEpisode == True:
                    hidden_state = None
                batch = test_loader.getbatch()
                if batch==False:
                    break
                x_test, y_test = batch
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                model.eval()
                yhat,hidden_state = model(x_test,hidden_state)
                predictions.append(yhat.cpu().detach().numpy())
                values.append(y_test.cpu().detach().numpy())
                inputFeature.append(x_test.cpu().detach().numpy())
        
        else:
            n=0
            while True:
                if test_loader.newEpisode == True:
                    hidden_state = None
                    n+=1
                if n==(episodeN+1):
                    break
                batch = test_loader.getbatch()
                if batch==False:
                    break
                x_test, y_test = batch
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                model.eval()
                yhat,hidden_state = model(x_test,hidden_state)
                predictions.append(yhat.cpu().detach().numpy())
                values.append(y_test.cpu().detach().numpy())
                inputFeature.append(x_test.cpu().detach().numpy())
                
    return predictions, values, inputFeature

def plot_test_scatter(p,o,n=None,outFolder=None,saveFig=False,note='',title=''):
    fig, ax = plt.subplots(1, 1,figsize = (6,5))
    if n==None:
        x=np.array(o)
        y=np.array(p)
    else:
        x=np.array(o[n[0]:n[1]])
        y=np.array(p[n[0]:n[1]])
    plt.scatter(x, y, 
                  color='b',  label='')
    R2 = np.corrcoef(x, y)[0, 1] ** 2
    RMSE = (np.sum((x - y) ** 2) / len(y)) ** 0.5
    plt.text(0.05, 0.87, r'$R^2 $ = %.3f'%R2, transform=ax.transAxes,fontsize=16)
    plt.text(0.05, 0.80, r'$RMSE $ = %.3f'%RMSE, transform=ax.transAxes,fontsize=16)
    plt.text(0.05, 0.73, r'$n $ = %d'%len(x), transform=ax.transAxes,fontsize=16)
    plt.xlabel('Observed', fontsize=14)
    plt.ylabel('Predicted', fontsize=14)
    lim = np.max([np.max(x),np.max(y)])
    plt.plot(np.arange(0,np.ceil(lim)+1), np.arange(0,np.ceil(lim)+1), 'k', label='1:1 line')
    plt.xlim([0,lim])
    plt.ylim([0,lim])
    
    if saveFig:
        if n==None:
            plt.title('%s samples all'%title)
            fig.savefig('%s/test_scatter_%s.png'%(outFolder,note))
        else:
            plt.title('%s samples from %d to %d'%(title,n[0],n[1]))
            fig.savefig('%s/test_scatter_%d-%d_%s.png'%(outFolder,n[0],n[1],note))
    else:
        plt.title(title)

def plotScatterDense(x_, y_,alpha=1, binN=200, thresh_p = None,outFolder='',
                     saveFig=False,note='',title='',uplim=None,downlim=None,auxText = None,legendLoc=4,naive=False):
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
        
        tmp = stats.linregress(x, y)
        para = [tmp[0],tmp[1]]
        # para = np.polyfit(x, y, 1)   # can't converge for large dataset

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
    if naive:
        pass
    else:
        plt.plot(np.arange(0,np.ceil(uplim)+1), np.arange(0,np.ceil(uplim)+1), 'k', label='1:1 line')
        plt.xlim([downlim, uplim])
        plt.ylim([downlim, uplim])
    plt.xlabel('Observations',fontsize=16)
    plt.ylabel('Predictions',fontsize=16)
    # plt.legend(loc=1)  # 指定legend的位置,读者可以自己help它的用法
    if not legendLoc is None:
        if legendLoc==False:
            plt.legend(edgecolor = 'w',facecolor='w',fontsize=12)          
        else:
            plt.legend(loc = legendLoc, edgecolor = 'w',facecolor='w',fontsize=12, framealpha=0)
    plt.title(title, y=0.9)
    
    metric={}
    if len(y) > 1:
        R2 = np.corrcoef(x, y)[0, 1] ** 2
        RMSE = (np.sum((y - x) ** 2) / len(y)) ** 0.5
        metric['R2'] = R2
        metric['RMSE'] = RMSE
        if naive:
            plt.text(0.05, 0.83,  r'$R^2 $= ' + str(R2)[:5],transform=ax.transAxes, fontsize=14)
            # ax = plt.text(0.1 * uplim, 0.86 * uplim, r'$MAPE $= ' + str(MAPE)[:5] + '%', fontsize=14)
            plt.text(0.05, 0.76,  r'$RMSE $= ' + str(RMSE)[:5],transform=ax.transAxes, fontsize=14)
        
            plt.text(0.05, 0.69, r'$Slope $= ' + str(para[0])[:5],  transform=ax.transAxes,fontsize=14)
            # plt.text(0.05, 0.62, r'$n $= ' + str(len(y)),  transform=ax.transAxes,fontsize=14)
        else:
            # MAPE = 100 * np.sum(np.abs((y - x) / (x+0.00001))) / len(x)
            plt.text(downlim + 0.1 * figRange, downlim + 0.83 * figRange, r'$R^2 $= ' + str(R2)[:5], fontsize=14)
            # ax = plt.text(0.1 * uplim, 0.86 * uplim, r'$MAPE $= ' + str(MAPE)[:5] + '%', fontsize=14)
            plt.text(downlim + 0.1 * figRange, downlim + 0.76 * figRange, r'$RMSE $= ' + str(RMSE)[:5], fontsize=14)
        
            plt.text(downlim + 0.1 * figRange, downlim + 0.69 * figRange, r'$Slope $= ' + str(para[0])[:5], fontsize=14)
            # plt.text(downlim + 0.1 * figRange, downlim + 0.62 * figRange, r'$n $= ' + str(len(y)),fontsize=14)
    if not auxText == None:
        plt.text(0.05, 0.9, auxText, transform=ax.transAxes,fontproperties = 'Times New Roman',fontsize=20)
    plt.colorbar()
    
    if saveFig:
        plt.title('%s samples all'%title)
        fig.savefig('%s/test_scatter_%s.png'%(outFolder,note))

    else:
        plt.title(title,fontsize=16)
         
    return metric

            
def plot_test_series(p,o,n=None,outFolder=None,saveFig=False,note='',title=''):
    if n==None:
        x=np.array(o)
        y=np.array(p)
    else:
        x=np.array(o[n[0]:n[1]])
        y=np.array(p[n[0]:n[1]])      
    fig = plt.figure(figsize=(10,5))
    plt.plot(y, 
                  color='r',  label='predicted')
    plt.plot(x,
                  color='y',  label='observaed')
    plt.legend()
    if saveFig:  
        if n==None:
            plt.title('%s samples all'%title)
            fig.savefig('%s/test_series_%s.png'%(outFolder,note))
        else:
            plt.title('%s samples from %d to %d'%(title,n[0],n[1]))
            fig.savefig('%s/test_series_%d-%d_%s.png'%(outFolder,n[0],n[1],note))
    else:
        plt.title(title)

def plot_test_series_compare(pList,oList,labelList,n=None,outFolder=None,saveFig=False,note='',title='',ylabel=''):
    color_list = ['darkred','r','darkgreen','g','m','b','k','y','c','sienna','navy','grey']
    
    maxV = np.max(np.stack([np.array(pList),np.array(oList)]))
    minV = np.min(np.stack([np.array(pList),np.array(oList)]))
    span = maxV-minV
    
    # pb model
    plt.figure(figsize=(10,5))    
    for i,o in enumerate(oList):
        label = labelList[i]
        if n==None:
            x=np.array(o)
        else:
            x=np.array(o[n[0]:n[1]])
     
        plt.plot(x, color=color_list[i],  label='%s %s ecosys model'%(title,label))
    plt.legend(fontsize=14)
    plt.xlabel('Day',fontsize=16)
    plt.ylabel(ylabel,fontsize=16)
    plt.title(title,fontsize=16)
    plt.ylim(minV-span*0.05,maxV+span*0.05)
    
    # net
    plt.figure(figsize=(10,5))    
    for i,p in enumerate(pList):
        label = labelList[i]
        if n==None:
            y=np.array(p)
        else:
            y=np.array(p[n[0]:n[1]])      
        
        plt.plot(y, color=color_list[i],  label='%s %s surrogate'%(title,label))

    plt.legend(fontsize=14)
    plt.xlabel('Day',fontsize=16)
    plt.ylabel(ylabel,fontsize=16)
    plt.title(title,fontsize=16)
    plt.ylim(minV-span*0.05,maxV+span*0.05)
    
def inperiod(d,periodDate):
    if (d>=periodDate[0]) & (d<=periodDate[1]):
        return True
    else:
        return False
    
def plot_test_series_en_compare(sList,labelList,fmtList,outFolder=None,saveFig=False,note='',title='',dateCol = None, scale = 1.0, periodDate=None):
    color_list = ['darkred','r','darkgreen','g','m','b','k','y','c','sienna','navy','grey']
    sList=[np.array(t).astype(float) for t in sList]
       
    fig = plt.figure(figsize=(13,5))
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
                        plt.plot_date(x,s[t,:],color = color_list[i] , linestyle='-', fmt='None', alpha=.10,linewidth=0.5, label=labelList[i])
                    else:
                        plt.plot_date(x,s[t,:],color = color_list[i] , linestyle='-', fmt='None', alpha=.10,linewidth=0.5)
    plt.legend()
    plt.xlabel('Day',fontsize = 13)
    plt.ylabel(title,fontsize = 13)
    plt.title(title)
    if saveFig:  
        fig.savefig('%s/test_series_%s.png'%(outFolder,note))

class RSobs():
    def __init__(self,data_dir,fileID = 'GPPextracted_25296_tile'):
        self.data_dir = data_dir
        self.fileID = fileID
        self.loadGPPdata()
        self.statistic()
        
    def loadGPPdata(self):
        hList = [10,11]
        vList = [4,5]
        dataList = []
        n=0
        pklFile = '%s/GPPextracted_25296.pkl'%self.data_dir
        if os.path.exists(pklFile):
            self.dataAll = load_object(pklFile)
        else:
            for h in hList:
                for v in vList:
                    outFile = '%s/%s_h%2dv%02d.csv'%(self.data_dir,self.fileID,h,v)
                    tmp = pd.read_csv(outFile)
                    tmp.drop(['Unnamed: 0'],axis=1,inplace=True)
                    if n>0:
                        tmp.drop(['Date'],axis=1,inplace=True)
                    n+=1
                    dataList.append(tmp)
            self.dataAll = pd.concat(dataList,axis=1)
            self.dataAll['Date'] = pd.to_datetime(self.dataAll['Date'],format='%Y-%m-%d')
            save_object(self.dataAll, pklFile)
        print('RS data loaded!')
    
    def getObs(self, site):
        obsDate = self.dataAll['Date'].tolist()
        obs = self.dataAll[site].tolist()
        return obsDate, obs
    
    def getObsYear(self, site, year):
        tmp = pd.DataFrame()
        tmp['Year'] = self.dataAll['Date'].dt.year
        tmp['Date'] = self.dataAll['Date']
        tmp[site] = self.dataAll[site]
        obs = tmp[tmp['Year']==year]
        return obs
    
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
            
    def plotGPP(self,CDL,DOYrange=[151,243]):
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
  
            GPP_sum_mean_corn.append(np.mean(np.sum(GPP_corn_year[DOYrange[0]:DOYrange[1]],axis=0)))
            GPP_sum_mean_soybean.append(np.mean(np.sum(GPP_soybean_year[DOYrange[0]:DOYrange[1]],axis=0)))
            # GPP_max_mean_corn.append(np.mean(np.max(GPP_corn_year,axis=0)))
            # GPP_max_mean_soybean.append(np.mean(np.max(GPP_soybean_year,axis=0)))
            GPP_max_mean_corn.append(np.mean(np.mean(np.sort(GPP_corn_year,axis=0)[-20:,:])))
            GPP_max_mean_soybean.append(np.mean(np.mean(np.sort(GPP_soybean_year,axis=0)[-20:,:])))
            
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
        cornFile = glob.glob('%s/*corn*.csv'%self.yieldPath)[0]
        soybeanFile = glob.glob('%s/*soy*.csv'%self.yieldPath)[0]
        self.yield_NASS_corn = pd.read_csv(cornFile)
        self.yield_NASS_soybean = pd.read_csv(soybeanFile)

def scatterDf(df_NASS,df_p, coef = 1, title='',saveFig=False,outFolder=None,note=''):
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
    
    x_,y_ = removeNaN(obs,pre,coef=coef) 
    metric = plotScatterDense(x_=x_, y_=y_, binN=100 ,title=title,saveFig=saveFig,outFolder=outFolder,note='all_%s'%note,naive=True)

    summary = {}
    summary['obs'] = dic_obs
    summary['pre'] = dic_pre
    summary['metric'] = metric

    return summary

def scatterYield(df_NASS,df_p, coef = 1, title='',saveFig=False,outFolder=None,note='',ave=True):
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
    
    difference = dic_pre.reset_index(drop=True)-dic_obs.reset_index(drop=True)
    difference['Year'] = YearList
    x_,y_ = removeNaN(obs,pre,coef=coef) 
    metric = plotScatterDense(x_=x_, y_=y_, binN=100 ,title=title,saveFig=saveFig,outFolder=outFolder,note='all_%s'%note)

    if ave:
        x_,y_ = removeNaN(obs_yearMean,pre_yearMean,coef=coef) 
        metric_ave = plotScatterDense(x_=x_, y_=y_, binN=100, title='Multi year mean %s'%title,
                              saveFig=saveFig,outFolder=outFolder,note='multiYearAve_%s'%note)
    else:
        metric_ave = None
    summary = {}
    summary['diff'] = difference
    summary['obs'] = dic_obs
    summary['pre'] = dic_pre
    summary['metric'] = metric
    summary['metric_ave'] = metric_ave
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
    
def NASS_vs_ecosys(df_NASS,df_p,crop,coef=1,outFolder=None,mode=None,saveFig=False,ave=True):
    year = df_NASS['Year'].tolist()
    validFIPS = list(set(df_NASS.columns[1:]).intersection(set(df_p.columns[1:])))
    validFIPS.sort()
    
    # corn
    NASS = np.array(df_NASS[validFIPS])
    ecosys = np.array(df_p[validFIPS])* coef
    NASS,ecosys=removeNaNmean(NASS,ecosys)
    if ave:
        fig = plt.figure(figsize=(12,5))
        plt.plot(year,NASS,'r-',marker='.',label='NASS_corn')
        plt.plot(year,ecosys,'g-',marker='*',label='ecosys_corn')
        plt.ylabel('Yield BU/Acre',fontsize=14)
        plt.xlabel('Year',fontsize=14)
    
    if saveFig:
        fig.savefig('%s/yield_3I_trend_%s_%s.png'%(outFolder,crop,mode))
    # scatter plot
    summary = scatterYield(df_NASS, df_p, 
                 coef = coef, title='KGML-DA vs. NASS, %s yield (BU/acre) %s'%(crop,mode),
                 saveFig=saveFig,outFolder=outFolder,note = '%s_%s'%(crop,mode),ave=ave)
    return summary

def NASS_vs_ecosys_trend(df_NASS,df_p,crop,coef=1,mode=None):
    # cal intersection
    validFIPS = list(set(df_NASS.columns[1:]).intersection(set(df_p.columns[1:])))
    validFIPS.sort()
    
    # obs and pre
    obs = []
    pre = []

    dic_obs = {}
    dic_pre = {}
    YearList = df_NASS['Year'].tolist()
    dic_obs['Year'] = YearList
    dic_pre['Year'] = YearList
    
    for FIPS in validFIPS:
        obs.extend(df_NASS[FIPS].tolist())
        pre.extend(df_p[FIPS].tolist())
        dic_obs[FIPS] = df_NASS[FIPS]/coef
        dic_pre[FIPS] = df_p[FIPS]
    
    dic_obs = pd.DataFrame(dic_obs)
    dic_pre = pd.DataFrame(dic_pre)
    
    difference = dic_pre.reset_index(drop=True)-dic_obs.reset_index(drop=True)
    difference['Year'] = YearList
    x_,y_ = removeNaN(obs,pre,coef=coef)
    
    metric = {}
    R2 = np.corrcoef(x_, y_)[0, 1] ** 2
    RMSE = (np.sum((y_ - x_) ** 2) / len(y_)) ** 0.5
    Bias = np.mean(y_) - np.mean(x_)
    metric['R2'] = R2
    metric['RMSE'] = RMSE
    metric['Bias'] = Bias
    
    summary = {}
    summary['diff'] = difference
    summary['obs'] = dic_obs
    summary['pre'] = dic_pre
    summary['metric'] = metric
    
    return summary

def validGPPcounty(GPPpath = 'F:/MidWest_counties/GPP',yearSpan = [t for t in range(2000,2020+1)]):
    fileName = 'validFIPS_GPP_cornBelt.pkl'
    if os.path.exists(fileName):
        validFIPS_GPP,nanYear_dic = load_object(fileName)
        return validFIPS_GPP,nanYear_dic
    
    countyPathes = glob.glob('%s/*.pkl'%GPPpath)
    FIPSList = list(set([t.split('_')[-2] for t in countyPathes]))
    FIPSList.sort()
    
    # check if anydata is missing
    validFIPS_GPP = []
    nanYear_dic = {}
    for n,FIPS in enumerate(FIPSList):
        GPP_corn = load_object('%s/GPP_%s_corn.pkl'%(GPPpath,FIPS))
        GPP_soybean = load_object('%s/GPP_%s_soybean.pkl'%(GPPpath,FIPS))
        nanYear_dic[FIPS] = [y for y in yearSpan if (np.isnan(GPP_corn[y][0])) | (np.isnan(GPP_soybean[y][0]))]
        if len(nanYear_dic[FIPS]) == 0:
            validFIPS_GPP.append(FIPS)
    save_object([validFIPS_GPP,nanYear_dic], fileName)        
    return validFIPS_GPP,nanYear_dic

def cornBeltCounty(path = 'F:/MidWest_counties/GPP'):
 
    countyPathes = glob.glob('%s/*.pkl'%path)
    FIPSList = list(set([t.split('_')[-2] for t in countyPathes]))
    FIPSList.sort()
      
    return FIPSList

def cornBeltCountyDA():
 
    dataRoot = r'F:/MidWest_counties/inputMerged_DA_countyMerge'
    countyPathes = glob.glob('%s/*.pkl'%dataRoot)
    FIPSList = [t.split('\\')[-1].split('_')[0] for t in countyPathes]
      
    return FIPSList

def FIPS_3I():
    dataRoot = 'F:/County_level_Dataset_3I/inputMerged_DA_merge'
    countyPathes = glob.glob('%s/*.pkl'%dataRoot)
    FIPSList_all = [t.split('\\')[-1].split('_')[0] for t in countyPathes]
    GPPpath = 'F:/MidWest_counties/GPP'
    
    # pick valid FIPS
    FIPSList = []
    for f in FIPSList_all:
        if os.path.exists('%s/GPP_%s_corn.pkl'%(GPPpath,f)) & os.path.exists('%s/GPP_%s_soybean.pkl'%(GPPpath,f)):
            FIPSList.append(f)
    
    return FIPSList

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
                elif (mode=='default')|(mode=='defaultV2'):
                    midYear = 2010
                    
                tmp = np.array(yield_df[yield_df['Year']==year])[:,1:]*coef+(year-midYear)*delta_y
            recorrected.append(tmp)
    keys = yield_df.columns.tolist()[1:]
    df_new = pd.DataFrame(np.concatenate(recorrected,axis=0),columns=keys)
    df_new.insert(0,'Year',yearRange_t)        
                    
    return df_new, yield_df
