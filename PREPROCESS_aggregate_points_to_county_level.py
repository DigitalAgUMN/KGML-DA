# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 13:07:45 2022

@author: yang8460

note: do not seperate corn and soybean for county-level climate and soil data
"""

import numpy as np
import matplotlib.pyplot as plt

import ECONET_util as util
import os, glob
import ECONET_dataPrepare_WarmingUp as E_data
import pandas as pd

def loadData(dataRoot,site,year):
    # load input and Ecosys output data
    inputData = '%s/%s_inputMerged.pkl'%(dataRoot,site)
    dataAll = E_data.load_object(inputData)
    yearList = [t.iloc[0]['Year'] for t in dataAll]
    loc = yearList.index(year)
    inputData = dataAll[loc]
    
    # obs
    
    return inputData

if __name__ == '__main__':
    
    # path  
    dataRoot = 'F:/MidWest_counties/inputMerged_DA'
    siteFile = 'F:/MidWest_counties/site_info_20299.csv'
    FIPS_dic,FIPSList,site_info = util.classifyPointsByCounty(siteFile)
    outPath = 'F:/MidWest_counties/inputMerged_DA_countyMerge'
    if not os.path.exists(outPath):
        os.makedirs(outPath)
        
    # input data
    yearList = [t for t in range(2000,2020+1)]
    GPPpath = 'F:/MidWest_counties/GPP'
    
    # 
    count = 0
    for n,FIPS in enumerate(FIPSList):
        FIPS_mean = []
        siteList = FIPS_dic[FIPS]
        
        for year in yearList:          
            countyYield_corn = []
    
            # load corn site data
            merge = []           
            for n_s,site in enumerate(siteList):            
                # load input and Ecosys output data            
                merge.append(loadData(dataRoot,site,year))
                                
            # average by county
            if count==0:
                keys = merge[0].columns
            if len(merge) >0:
                ave = pd.DataFrame(np.mean(np.array(merge),axis=0),columns=keys)
                FIPS_mean.append(ave)

            else:
                FIPS_mean.append(None)

            count+=1
    
        # save
        E_data.save_object(obj=FIPS_mean, filename='%s/%s_inputMerged.pkl'%(outPath,FIPS))
        print('%d/%d,county %s have %s sites.'%(n+1,len(FIPSList),FIPS,len(siteList)))