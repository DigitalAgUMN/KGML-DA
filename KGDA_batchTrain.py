# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 12:49:28 2022

@author: yang8460

This script is used for training a KGDA surrogate. The structure of this surrogate refers to https://doi.org/10.1016/j.rse.2023.113880
"""

import torch
import torch.nn as nn
import torch.optim as optim
import KGDA_util as util
import KGDA_Networks as net
from scipy import stats

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data.dataloader import DataLoader
import os
import glob

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device}" " is available.")
 
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


            
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

def slope_np(x,y):
    xyMean = np.mean(x*y)
    xMean = np.mean(x)
    yMean = np.mean(y)
    x2Mean = np.mean(x**2)
    return (xyMean-xMean*yMean)/(x2Mean-xMean**2 + 10e-6)

def plotScatterDense(x_, y_,alpha=1, binN=200, thresh_p = 0.05,outFolder='',
                     saveFig=False,note='',title='',uplim=None,downlim=None,auxText = None,legendLoc=4):
        fig, ax = plt.subplots(1, 1,figsize = (7,5))
        x_=np.array(x_)
        y_=np.array(y_)
        if len(y_) > 1:
                       
            # Calculate the point density
            if not (thresh_p is None):
                thresh = (np.max(np.abs(x_))*thresh_p)
                
                x = x_[((x_>thresh)|(x_<-thresh)) & ((y_>thresh)|(y_<-thresh))]
                y = y_[((x_>thresh)|(x_<-thresh)) & ((y_>thresh)|(y_<-thresh))]
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

        plt.scatter(x2, y2, c=z2, cmap='Reds', marker='.',alpha=alpha)
        
        if uplim==None:
            uplim = 1.2*max(np.hstack((x, y)))
        if downlim==None:
            downlim = 0.8*min(np.hstack((x, y)))
            
        figRange = uplim - downlim
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

def plot_test_singleStep(pre,obs,n):
    plt.figure()
    plt.scatter(np.squeeze(obs)[:n], np.squeeze(pre)[:n], 
                  color='b',  label='')
    plt.xlabel('Observed', fontsize=14)
    plt.ylabel('Predicted', fontsize=14)
    plt.figure(figsize=(10,5))
    plt.plot(np.squeeze(pre)[:n], 
                  color='r',  label='predicted')
    plt.plot(np.squeeze(obs)[:n],
                  color='y',  label='observaed')
    plt.legend()
    
if __name__ == '__main__':
    
    # hyper parameters
    batch_size = 64#256
    hidden_dim = 64 # 64
    n_epochs = 10 #100
    learning_rate = 1e-3
    lr_decay = 0.96
    saveResult = True
            
    # inputs and outputs
    X_selectFeatures = ['Tair','RH','Wind','Precipitation','Radiation','GrowingSeason',
                        'CropType','VCMX','CHL4','GROUPX','STMX','GFILL','SLA1','Bulk density','Field capacity'
                        ,'Wilting point','Ks','Sand content','Silt content','SOC','Fertilizer'] 
    y_selectFeatures = ['DVS',
                        'ET_daily','GPP_daily','AVE_SM',
                        'Biomass','Reco_daily','NEE_daily',
                        'LAI','GrainYield']
    output_dim=[1,3,3,2]
    
    # normalization coef
    y_NormCoef = [0.5,
                  0.15, 0.02, 1.5,
                  0.001, 0.06, -0.05,
                  0.1,0.0015]
    
    # project information
    modelPath = 'model'
    note = 'writeNotesHere'
    now = datetime.now().strftime('%y%m%d-%H%M%S')
    projectName = 'epoch%d-batch%d-%s-%s'%(n_epochs,batch_size,note,now)
    outFolder = 'log/%s'%(projectName)
    if saveResult:     
        if not os.path.exists('log'):
            os.mkdir('log') 
        if not os.path.exists(outFolder):
            os.mkdir(outFolder)
        if not os.path.exists(modelPath):
            os.mkdir(modelPath)
            
    # load dataset 
    dataPath = r'demoData/surrogateTraining'
    siteYearList = [t.replace('\\','/').split('/')[-1].split('input_')[-1].split('.')[0] for t in glob.glob('%s/input_*.pkl'%dataPath)]
    inputList = ['%s/input_%s.pkl'%(dataPath,t) for t in siteYearList]
    outputList = ['%s/output_%s.pkl'%(dataPath,t) for t in siteYearList]
    
    # split dataset
    X_train, X_test, y_train, y_test = util.train_test_split_no_leak(X=inputList, y = outputList,
                                                                                  test_ratio=0.1)
    
    # get the dataloader ready  
    train_ds = util.EcoNet_dataset_pkl_yearly(X_train,y_train,X_selectFeatures=X_selectFeatures, y_selectFeatures=y_selectFeatures,y_NormCoef=y_NormCoef)
    test_ds = util.EcoNet_dataset_pkl_yearly(X_test,y_test,X_selectFeatures=X_selectFeatures, y_selectFeatures=y_selectFeatures,y_NormCoef=y_NormCoef)    
    train_dl = DataLoader(train_ds,batch_size=batch_size, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_ds,batch_size=batch_size, shuffle=True, num_workers=0)
    test_dl_one = DataLoader(test_ds, batch_size=1, shuffle=False)

    # get some data samples for debugging
    b0,b1,l = train_ds[1]
    for a in train_dl:
        break  
    
    ## initialize the network
    input_dim = len(X_selectFeatures)
    model = net.KG_ecosys(input_dim=input_dim,hidden_dim=hidden_dim,output_dim=output_dim)
    model = to_device(model, device)  
    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Decay LR by a factor every epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_decay)
    
    # train
    opt = util.Optimization_decoder(model=model, loss_fn=loss_fn, optimizer=optimizer,exp_lr_scheduler=exp_lr_scheduler, hiera=True)
    opt.train_yearly(train_dl, test_dl, n_epochs=n_epochs, n_features=input_dim, interv = 30, timeCount=True)
    
    # plot training losses
    opt.plot_losses(outFolder,saveFig=saveResult)
     
    # save results
    if saveResult:       
        torch.save(model.state_dict(), '%s/%s_state_dict.pth'%(modelPath,projectName))
        torch.save(model, '%s/%s.pth'%(modelPath,projectName))
        losses = pd.DataFrame()
        losses['train_losses'] = opt.train_losses
        losses['train_losses_main'] = opt.train_losses_main
        losses['train_losses_de'] = opt.train_losses_de
        
        losses['val_losses'] = opt.val_losses
        losses['val_losses_main'] = opt.val_losses_main
        losses['val_losses_de'] = opt.val_losses_de
        losses.to_csv('%s/train_losses_log.csv'%(outFolder))
        
    # test  
    predictions, values = opt.evaluate_yearly(test_dl_one, batch_size=1, n_features=input_dim)

    for i, sf in enumerate(y_selectFeatures):
        p = []
        _= [p.extend(np.squeeze(t[:,:,i])/y_NormCoef[i]) for t in predictions]
        o = []
        _= [o.extend(np.squeeze(t[:,:,i])/y_NormCoef[i]) for t in values]
        
        if saveResult:
            data = pd.DataFrame()
            data['predictions'] = p
            data['observations'] = o
            data.to_csv('%s/test_series_%s.csv'%(outFolder,sf))
    
        # plot_test_scatter(p=p,o=o,outFolder=outFolder,saveFig=saveResult,note='all_%s'%sf,title=sf)
        plotScatterDense(x_=o, y_=p,outFolder=outFolder,saveFig=saveResult,note='all_%s'%sf,title=sf)
        plot_test_series(p=p,o=o,n=None,outFolder=outFolder,saveFig=saveResult,note=sf,title=sf)

    if saveResult:        
        lengthList = [t.shape[1] for t in values]
        with open('%s/test_series_lengthList.txt'%(outFolder),'w') as f:
            _=[f.write('%d\n'%t) for t in lengthList]
        
