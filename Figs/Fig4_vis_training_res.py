# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 14:19:16 2023

@author: yang8460

Plot the training results for Fig.4 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
import random

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['figure.dpi'] = 300

def plotScatterDense(x_, y_,alpha=1, binN=200, thresh_p = 0.05,note='',title='',uplim=None,downlim=None,
                     auxText = None,legendLoc=4, cmap='Reds', removeNeg=False,
                     vmin=None,vmax=None,removeZero=True,upcoef=1.2,plotdense=False):
        ax = plt.subplot(3,3,i+1)
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
        
        if plotdense:
            # histogram the data
            hh, locx, locy = np.histogram2d(x, y, bins=bins)
    
            # Sort the points by density, so that the densest points are plotted last
            z = np.array([hh[np.argmax(a<=locx[1:]),np.argmax(b<=locy[1:])] for a,b in zip(x,y)])
            idx = z.argsort()
            x2, y2, z2 = x[idx], y[idx], z[idx]
    
            # plt.scatter(x2, y2, c=z2, cmap=cmap, marker='.',alpha=alpha)
            plt.scatter(x2, y2, c=z2, cmap=cmap, marker='.',alpha=alpha,vmin=vmin,vmax=vmax)
        else:
            plt.scatter(x, y)
            
        if uplim==None:
            uplim = upcoef*max(np.hstack((x, y)))
        if downlim==None:
            downlim = 0.8*min(np.hstack((x, y)))
            
        figRange = uplim - downlim
        plt.plot(np.arange(downlim-1,np.ceil(uplim)+1), np.arange(downlim-1,np.ceil(uplim)+1), 'k', label='1:1 line')
        plt.xlim([downlim, uplim])
        plt.ylim([downlim, uplim])
        # plt.xlabel('$Ecosys$ simulation',fontsize=16)
        # plt.ylabel('Surrogate predictions',fontsize=16)
        # plt.legend(loc=1)  # 指定legend的位置,读者可以自己help它的用法
        if not legendLoc is None:
            if legendLoc==False:
                plt.legend(edgecolor = 'w',facecolor='w',fontsize=12)          
            else:
                plt.legend(loc = legendLoc, edgecolor = 'w',facecolor='w',fontsize=12, framealpha=0)
        plt.title(title, y=0.9, fontsize=16)
        
        if len(y) > 1:
            R2 = np.corrcoef(x, y)[0, 1] ** 2
            RMSE = (np.sum((y - x) ** 2) / len(y)) ** 0.5
            NRMSE = ((np.sum((y - x) ** 2) / len(y)) ** 0.5)/np.mean(x)
            Bias = np.mean(y) - np.mean(x)
            NBias = np.abs((np.mean(y) - np.mean(x))/np.mean(x))
            # MAPE = 100 * np.sum(np.abs((y - x) / (x+0.00001))) / len(x)
            plt.text(downlim + 0.1 * figRange, downlim + 0.83 * figRange, r'$R^2 $= ' + str(R2)[:5], fontsize=14)
            # ax = plt.text(0.1 * uplim, 0.86 * uplim, r'$MAPE $= ' + str(MAPE)[:5] + '%', fontsize=14)
            plt.text(downlim + 0.1 * figRange, downlim + 0.76 * figRange, r'$RMSE $= ' + str(RMSE)[:5], fontsize=14)
            plt.text(downlim + 0.1 * figRange, downlim + 0.69 * figRange, r'$Bias $= ' + str(Bias)[:5], fontsize=14)
            
            # plt.text(downlim + 0.1 * figRange, downlim + 0.69 * figRange, r'$NRMSE $= ' + str(NRMSE)[:5], fontsize=14)
            plt.text(downlim + 0.1 * figRange, downlim + 0.62 * figRange, r'$Slope $= ' + str(para[0])[:5], fontsize=14)
            # plt.text(downlim + 0.1 * figRange, downlim + 0.55 * figRange, r'$Nbais $= ' + str(NBias)[:5], fontsize=14)
            # plt.text(downlim + 0.1 * figRange, downlim + 0.62 * figRange, r'$Bias $= ' + str(Bias)[:5], fontsize=14)
        if not auxText == None:
            plt.text(0.05, 0.91, auxText, transform=ax.transAxes,fontproperties = 'Times New Roman',fontsize=20)
        plt.colorbar()
        
      
            
if __name__ == '__main__':
    # pathTrainingRes = r'F:/OneDrive/ecosys_RNN/log/gru-epoch30-batch256-cornBelt20299_4cell_v1_2_RecoCorrected_paraPheno_c2-221010-000709'
    pathTrainingRes = 'path of your training output'
    
    termList = ['DVS',
                        'ET_daily','GPP_daily','AVE_SM',
                        'Biomass','Reco_daily','NEE_daily',
                        'LAI','GrainYield']
    
    titleList = ['DVS',
                        'ET (mm)','GPP (gC/$m^2$/day)','Soil moisture (0-30 cm)',
                        'Biomass (gC/$m^2$)','Reco (gC/$m^2$/day)','NEE (gC/$m^2$/day)',
                        'LAI ($m^2$/$m^2$)','GrainYield (gC/$m^2$)']
    
    fig = plt.figure(figsize=(13,10))
    numberList = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)']
    i = 0
    for term, title, number in zip(termList, titleList, numberList):
        
        data_df = pd.read_csv('%s/test_series_%s.csv'%(pathTrainingRes,term))
        
        # the original points are too many so we only plot 10% of the total points
        random.seed(0)
        randomIndex = random.sample(range(0,len(data_df)),int(len(data_df)*0.1))
        x = data_df['predictions'].iloc[randomIndex].tolist()
        y = data_df['observations'].iloc[randomIndex].tolist()
    
        if term == 'NEE_daily':
            thresh_p = None
        else:
            thresh_p = 0.02
 
        vmin=0
        vmax=500
        if (term == 'ET_daily') | (term=='GPP_daily') | (term=='LAI'):
            removeNeg = True
        else:
            removeNeg = False
        if term == 'DVS':
            upcoef = 1.2
        else:
            upcoef = 1.0
            
        plotScatterDense(x_=x, y_=y,note='%s_500'%term,thresh_p=thresh_p, title=title, cmap='gist_heat_r',
                          auxText=number, removeNeg=removeNeg,
                          vmin=vmin, vmax=vmax,upcoef=upcoef,plotdense=True)#cmap='hot_r'
        
        
        i+=1
        print('%s is finished'%term)

    fig.tight_layout(pad=0)
    fig.text(0.5, -0.03, '$Ecosys$ simulation', ha='center',fontsize=22)
    fig.text(-0.04, 0.5, 'Surrogate prediction', va='center', rotation='vertical',fontsize=22)
    # fig.savefig('test.png')