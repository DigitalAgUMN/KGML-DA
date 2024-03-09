# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 23:42:55 2022

@author: yang8460
"""

import pandas as pd
import numpy as np
import glob
import copy
import os
import sys
import datetime
import shutil
version = int(sys.version.split()[0].split('.')[1])
if version > 7:
    import pickle
else:
    import pickle5 as pickle

def reduceDigtal(data):
    data_reduced = []
    for row in data:
        tmp = []
        for t in row:
            tmp.append('%.3f'%t)
        data_reduced.append(tmp)
    return data_reduced

def loadPara(filename,smethod='space'):
    with open(filename, 'r') as f:
            data_o = f.readlines()
    if smethod=='space':
        data = [t.strip().split() for t in data_o]
        
    elif smethod=='comma':
        data = [t.strip().split(',') for t in data_o]
        
    if ',' in data[0][0]:
        data = [t.strip().split(',') for t in data_o]
    return data

def paraStatistic(ParaALL,nanLoc=None):
    paraStat = []
    template = []
    for rows in range(len(ParaALL[0])):
        item = []
        item_template=[]
        for cols in range(len(ParaALL[0][rows])):
            tmp = []
            for i,t in enumerate(ParaALL):
                if not (nanLoc is None):
                    if i in nanLoc:
                        continue
                tmp.append(np.float32(t[rows][cols]))
            item.append({'range':(np.min(tmp),np.max(tmp)),'mean':np.mean(tmp),'std':np.std(tmp)})
            item_template.append(np.mean(tmp))
        paraStat.append(item)
        template.append(item_template)
   
    return paraStat,template
    
def hour2Day(data, method='mean'):
    dayList = []
    i=0
    tmp = []
    for t in data:
        if method=='radiation':
            if i<23:
                tmp.append(t*3600/1e6)
                i+=1
            else:
                dayList.append(np.sum(tmp))
                i=0
                tmp = []
        else:
            if i<23:
                tmp.append(t)
                i+=1
            else:
                if method=='mean':
                    dayList.append(np.mean(tmp))
                elif method=='accum':
                    dayList.append(np.sum(tmp))
                elif method=='max':
                    dayList.append(np.max(tmp))
                elif method=='min':
                    dayList.append(np.min(tmp))
                elif method=='date':
                    dayList.append(tmp[-1])
                else:
                    raise ValueError
                i=0
                tmp = []
    return dayList

def day2year(df,target='GrainYield'):
    doy = df.DOY
    endLoc = []
    for i in range(len(doy)):
        if i==0:
            pass
        else:
            if doy[i]<doy[i-1]:
                endLoc.append(i-2)
    df_new = pd.DataFrame(columns=df.keys())
    for t in df.keys():
        df_new[t] = df[t][endLoc]  
    # df_new['Date'] = df_new['Date'].astype(int).astype(str).str.zfill(8)
    return df_new

def loadWeather(weatherList,df_w):
    df_w_d = pd.DataFrame()
    for w in weatherList:
        df_w_d[w[0]]=hour2Day(df_w[w[1]], method=w[2])

    return df_w_d

def gSURRGO_temp(soilPath,s):
    parafileALL = glob.glob('%s/mesoil_site_%s_*'%(soilPath,s.split('_')[0]))
    ParaALL_soil = [loadPara(t) for t in parafileALL]
    
    soilAlbedo = [t[0][2] for t in ParaALL_soil]
    bulkDens = [t[2][0] for t in ParaALL_soil]
    nanLoc_albedo = [i for i,t in enumerate(soilAlbedo) if t=='nan']
    nanLoc_bulk = [i for i,t in enumerate(bulkDens) if t=='nan']
    # check the equality for crop file
    gSURRGO,gSURRGO_template = paraStatistic(ParaALL_soil,nanLoc=nanLoc_albedo+nanLoc_bulk)
    gSURRGO_template = reduceDigtal(gSURRGO_template)
    
    return gSURRGO_template

def loadSoil(soilPath,s,soilParaList):
    
    mesoil = loadPara('%s/mesoil_site_%s'%(soilPath,s))
    tmp = []
    for t in mesoil:
        tmp.extend(t)
    if 'nan' in tmp:
        mesoil=gSURRGO_temp(soilPath,s)
        print('nan in soil file of %s, use template'%(s))
        
    df_s = pd.DataFrame(columns=[t[0] for t in soilParaList])
    df_s.loc[0] = [np.mean(np.array(mesoil[t[2]][t[3][0]:t[3][1]]).astype(np.float32)) for t in soilParaList]       

    return df_s

def mergeInOut(df_w_d,df_s):
    
    w = df_w_d
    yearList = list(set(w['Year']))
    yearList.sort()
    
    ## slice data by year  
    inputMerged = []
    for y in yearList:
        # slice weather
        tmp = copy.deepcopy(w[w.Year==y])
        for t in selectFeatures[0]:
            tmp[t] = 0
        # add soil properties
        for t in selectFeatures[1]:
            tmp[t] = copy.deepcopy(df_s[t].item())
        for t in selectFeatures[2]:
            tmp[t] = 0
        inputMerged.append(tmp)
        
    return inputMerged

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)    

if __name__ == '__main__':
    
    # data path
    # weatherPath = r'E:\County_level_Dataset_3I\site'
    # soilPath = r'E:\County_level_Dataset_3I\gSSURGO_Ecosys'
    # # outPath =  r'E:\County_level_Dataset_3I\inputMerged_DA'
    # outPath =  r'F:\County_level_Dataset_3I\inputMerged_DA_v2'  # Tmin, Tmax provided
    home = r'F:/MidWest_counties'
    weatherPath = '%s/site'%home
    soilPath = '%s/gSSURGO_Ecosys'%home
    outPath =  '%s/inputMerged_DA'%home
    Tminmax = False
    
    if not os.path.exists(outPath):
        os.makedirs(outPath)
        
    # site info
    site_info = pd.read_csv('%s/site_info_20299.csv'%home)
    siteID = site_info['Site_ID'].tolist()
    weatherMapping = pd.read_csv('%s/site/mapping.csv'%home)
    siteID_w = weatherMapping['Mapping_to_unique'].tolist()
       
    if Tminmax:
        weatherList = [('Year','Year','date'),('Day','DOY','date'),('Tair','tair','mean'),('TairMax','tair','max')
                       ,('TairMin','tair','min'),('RH','spRH','mean'),
                       ('Wind','wind','mean'),('Precipitation','precip','accum'),
                       ('Radiation','swdown','radiation')]
    else:
        weatherList = [('Year','Year','date'),('Day','DOY','date'),('Tair','tair','mean'),('RH','spRH','mean'),
                        ('Wind','wind','mean'),('Precipitation','precip','accum'),
                        ('Radiation','swdown','radiation')]
    
    selectFeatures = [['GrowingSeason',
                        'CropType','VCMX','CHL4','GROUPX','STMX','GFILL','SLA1'],['Bulk density','Field capacity'
                        ,'Wilting point','Ks','Sand content','Silt content','SOC'],['Fertilizer']] 
    
    # soil data, average of 0-3 layer, that is, 0.01,0.05,0.15,and 0.3
    soilParaList = [('Bulk density',5,2,[0,3]),('Field capacity',5,3,[0,3]),
                 ('Wilting point',5,4,[0,3]),('Ks',5,5,[0,3]),('Sand content',5,7,[0,3]),
                 ('Silt content',5,8,[0,3]),('SOC',5,14,[0,3])]
    
    n = 0
    pool = {}
    for s,s_w in zip(siteID,siteID_w):
                
        # load weather data
        if not(s_w in list(pool.keys())):
            df_w = pd.read_csv('%s/%s.csv'%(weatherPath,s_w))
            df_w_d = loadWeather(weatherList,df_w)
            pool[s_w] = copy.deepcopy(df_w_d)
        else:
            df_w_d = copy.deepcopy(pool[s_w])
            print('%s is buffered'%s)
        
        # load soil data
        df_s = loadSoil(soilPath,s,soilParaList)
        
        # merge
        inputMerged = mergeInOut(df_w_d,df_s)
        save_object(inputMerged, '%s/%s_inputMerged.pkl'%(outPath,s))
        n+=1
        
        print('processed %s, %s sites finished'%(s,n))