# -*- coding: utf-8 -*-
"""
Created on Tue May 10 16:29:11 2022

@author: yang8460
"""

import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['figure.dpi'] = 300

def plot_test_series(sList,labelList, n=None,outFolder=None,saveFig=False,note='',title='',dateCol = None, scale = 1.0, discrete = False, lowLim=-999):
    color_list = ['k','r','y','g','b','c','m','sienna','navy','grey']
    if n==None:
        sList=[np.array(t) for t in sList]
    else:
        sList=[np.array(t[n[0]:n[1]]) for t in sList]
       
    fig = plt.figure(figsize=(13,5))
    for i,s in enumerate(sList):
        s[s<lowLim]=np.nan
        if dateCol is None:
            plt.plot(s*scale, color=color_list[i],  label=labelList[i])
        else:
            if n==None:
                x=dateCol[i]
            else:
                x=dateCol[i][n[0]:n[1]]
            if discrete:
                plt.plot_date(x,s*scale, color=color_list[i],  label=labelList[i], fmt='*')
            else:
                plt.plot_date(x,s*scale, color=color_list[i],  label=labelList[i], linestyle='-' ,fmt='None')
 
    plt.legend()
    plt.xlabel('Day')
    plt.ylabel(title)
    
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
                    dayList.append(int(str(tmp[-1])[:8]))
                else:
                    raise ValueError
                i=0
                tmp = []
    return dayList
    
def loadHourly(termList,source):      
    df_w = pd.DataFrame()
    for w in termList:
        df_w[w[0]]=hour2Day(source[w[1]], method=w[2])
        
    return df_w
    
def read_flux(siteList, period):
    # define dataframes
    NEE_data = pd.DataFrame(columns=['Date'] + siteList)
    GPP_data = pd.DataFrame(columns=['Date'] + siteList)
    Reco_data = pd.DataFrame(columns=['Date'] + siteList)
    
    Rn_data = pd.DataFrame(columns=['Date'] + siteList)
    LE_data = pd.DataFrame(columns=['Date'] + siteList)
    H_data = pd.DataFrame(columns=['Date'] + siteList)
    G_data = pd.DataFrame(columns=['Date'] + siteList)
    energyBalance = pd.DataFrame(columns=['Date'] + siteList)
    ET_data = pd.DataFrame(columns=['Date'] + siteList)
    SWC_data = pd.DataFrame()
    SWC_data_10cm_AVE = pd.DataFrame()
    SWC_data_30cm_AVE = pd.DataFrame()
    
    # read data
    for i,site in enumerate(siteList):
        dataFolder_flux = glob.glob('demoData/siteData/FluxNet/*%s_FLUXNET2015_SUBSET*'%site)
        dataPath_flux = glob.glob('%s/*%s_FLUXNET2015_SUBSET_DD*'%(dataFolder_flux[0],site))[0]
        dataFolder_ameriflux = glob.glob('demoData/siteData/AmeriFlux/*%s_BASE-BADM*'%site)
        dataPath_ameriflux = glob.glob('%s/*%s_BASE*'%(dataFolder_ameriflux[0],site))[0]
        
        # extract data inside AmeriFlux with the target period
        tmp_ameriflux = pd.read_csv(dataPath_ameriflux,skiprows=2)
        termList = [('Date','TIMESTAMP_START','date'),
                    ('SWC_2cm','SWC_PI_F_1_1_1','mean'),
                    ('SWC_5cm','SWC_PI_F_1_2_1','mean'),
                    ('SWC_10cm','SWC_PI_F_1_3_1','mean'),
                    ('SWC_30cm','SWC_PI_F_1_4_1','mean')]
        dailyAmeriFlux = loadHourly(termList,source=tmp_ameriflux)
        locS_A = int(np.array(dailyAmeriFlux[dailyAmeriFlux.Date == int(period[0])].index))
        locE_A = int(np.array(dailyAmeriFlux[dailyAmeriFlux.Date == int(period[1])].index))
        if i==0:
            SWC_data['Date'] = dailyAmeriFlux['Date'].loc[locS_A:locE_A+1]
            SWC_data_10cm_AVE['Date'] = dailyAmeriFlux['Date'].loc[locS_A:locE_A+1]
            SWC_data_30cm_AVE['Date'] = dailyAmeriFlux['Date'].loc[locS_A:locE_A+1]
        for t in termList[1:]: 
            SWC_data['%s_%s'%(site,t[0])] = dailyAmeriFlux[t[0]].loc[locS_A:locE_A+1]
        SWC_data_10cm_AVE[site] = np.mean(dailyAmeriFlux[['SWC_2cm','SWC_5cm','SWC_10cm']].loc[locS_A:locE_A+1].values,axis=1)
        SWC_data_30cm_AVE[site] = np.mean(dailyAmeriFlux[['SWC_2cm','SWC_5cm','SWC_10cm','SWC_30cm']].loc[locS_A:locE_A+1].values,axis=1)
        
        # extract data inside FluxNet withthe target period
        tmp = pd.read_csv(dataPath_flux)
        locS = int(np.array(tmp[tmp.TIMESTAMP == int(period[0])].index))
        locE = int(np.array(tmp[tmp.TIMESTAMP == int(period[1])].index))
        if i==0:
            NEE_data['Date'] = tmp['TIMESTAMP'].loc[locS:locE+1]
            GPP_data['Date'] = tmp['TIMESTAMP'].loc[locS:locE+1]
            Reco_data['Date'] = tmp['TIMESTAMP'].loc[locS:locE+1]
            
            Rn_data['Date'] = tmp['TIMESTAMP'].loc[locS:locE+1]
            LE_data['Date'] = tmp['TIMESTAMP'].loc[locS:locE+1]
            H_data['Date'] = tmp['TIMESTAMP'].loc[locS:locE+1]
            G_data['Date'] = tmp['TIMESTAMP'].loc[locS:locE+1]
            energyBalance['Date'] = tmp['TIMESTAMP'].loc[locS:locE+1]
            ET_data['Date'] = tmp['TIMESTAMP'].loc[locS:locE+1]
            
        NEE_data[site] = tmp['NEE_VUT_REF'].loc[locS:locE+1]
        GPP_data[site] = tmp['GPP_NT_VUT_REF'].loc[locS:locE+1]
        Reco_data[site] = tmp['RECO_NT_VUT_REF'].loc[locS:locE+1]
        
        Rn_data[site] = tmp['NETRAD'].loc[locS:locE+1]
        LE_data[site] = tmp['LE_CORR'].loc[locS:locE+1]
        H_data[site] = tmp['H_CORR'].loc[locS:locE+1]
        G_data[site] = tmp['G_F_MDS'].loc[locS:locE+1]
        energyBalance[site] = tmp['NETRAD'].loc[locS:locE+1] -tmp['G_F_MDS'].loc[locS:locE+1] \
            - tmp['LE_CORR'].loc[locS:locE+1] -tmp['H_CORR'].loc[locS:locE+1]
        ET_data[site] = tmp['LE_CORR'].loc[locS:locE+1]*0.035  # 0.035 is coef from w/m2 to mm
        
        NEE_data['Date'] = pd.to_datetime(NEE_data['Date'],format='%Y%m%d')
        GPP_data['Date'] = pd.to_datetime(GPP_data['Date'],format='%Y%m%d')
        Reco_data['Date'] = pd.to_datetime(Reco_data['Date'],format='%Y%m%d')
        energyBalance['Date'] = pd.to_datetime(energyBalance['Date'],format='%Y%m%d')
        ET_data['Date'] = pd.to_datetime(ET_data['Date'],format='%Y%m%d')
        SWC_data['Date'] = pd.to_datetime(ET_data['Date'],format='%Y%m%d')
        SWC_data_10cm_AVE['Date'] = pd.to_datetime(ET_data['Date'],format='%Y%m%d')
        SWC_data_30cm_AVE['Date'] = pd.to_datetime(ET_data['Date'],format='%Y%m%d')
        
    return NEE_data,GPP_data,Reco_data,energyBalance,ET_data,SWC_data,SWC_data_10cm_AVE,SWC_data_30cm_AVE
    
# def readItem(site, tmp, target):
#     item_index = loc_extracter(df=tmp,field='VARIABLE_GROUP',target=target)
#     item_df_vec = tmp.loc[item_index]
#     varList = list(set(item_df_vec['VARIABLE'].tolist()))
#     item_df = pd.DataFrame(columns=['site']+varList)
#     for n,var in enumerate(varList):
#         var_index = loc_extracter(df=item_df_vec,field='VARIABLE',target=var)
#         if n==0:
#             item_df[var] = item_df_vec.loc[var_index]['DATAVALUE'].tolist()
#             length = len(var_index)
#         else:
#             if length==len(var_index):
#                 item_df[var] = item_df_vec.loc[var_index]['DATAVALUE'].tolist()
#             else:
#                 print('%s: size not match, discard'%var)
#     item_df['site'] = site
    
#     return item_df

class readData():
    def __init__(self,siteList):
        self.siteList = siteList
        self.dataList = [None for t in self.siteList]

    def read_item(self,target,dateSort,discardTerm=None):
        df_list = []
        for i,site in enumerate(self.siteList):
            dataFolder = glob.glob('demoData/siteData/AmeriFlux/*%s_BASE-BADM*'%site)
            dataPath = glob.glob('%s/*%s_BIF*'%(dataFolder[0],site))[0]
        
            # extract the target period
            if self.dataList[i] is None:
                self.dataList[i] = pd.read_excel(dataPath)
            
            df = self.readItem_id(site, self.dataList[i], target=target)
            if not (discardTerm is None):
                df = df[df[discardTerm[0]]!=discardTerm[1]]
            df[dateSort] = pd.to_datetime(df[dateSort])
            df = df.sort_values(by=dateSort)
            df_list.append(df)
 
        return df_list

    def readItem_id(self,site, tmp, target):
        item_index = self.loc_extracter(df=tmp,field='VARIABLE_GROUP',target=target)
        item_df_vec = tmp.loc[item_index]
        varList = ['GROUP_ID']+list(set(item_df_vec['VARIABLE'].tolist()))
        ID_list = list(set(item_df_vec['GROUP_ID']))
           
        item_df = pd.DataFrame(columns=['SITE_ID']+varList)
        ['SITE_ID']
        for n,id_g in enumerate(ID_list):
            tmp=item_df_vec[item_df_vec['GROUP_ID']==id_g].copy()
            item_df.loc[n] = 'No data'
            
            for i in range(tmp.shape[0]):
                for v in varList:          
                    if v in tmp.columns:
                        item_df.loc[n,v] = tmp.iloc[i][v]
                    else:
                        var_tmp = tmp.iloc[i]['VARIABLE']
                        item_df.loc[n,var_tmp] = tmp.iloc[i]['DATAVALUE']
    
        item_df['SITE_ID'] = site
            
        return item_df    

    def loc_extracter(self,df,field,target):
        indexList = df[field][df[field]==target].index
        return indexList
    
def read_BIF_id(siteList):
    
    D = readData(siteList)
    LAI_df_list = D.read_item(target='GRP_LAI',dateSort='LAI_DATE')
    
    # biomass
    biomass_df_list = D.read_item(target='GRP_AG_BIOMASS_CROP',dateSort='AG_BIOMASS_DATE',
                                  discardTerm=['AG_BIOMASS_CROP_ORGAN','Total'])
    
    # LMA
    LMA_df_list = D.read_item(target='GRP_LMA',dateSort='LMA_DATE')
    
    # production
    prod_df_list = D.read_item(target='GRP_AG_PROD_CROP',dateSort='AG_PROD_DATE_START')
    
    # planting
    planting_df_list = D.read_item(target='GRP_DM_PLANTING',dateSort='DM_DATE')
    
    # harvest
    harvest_df_list = D.read_item(target='GRP_DM_AGRICULTURE',dateSort='DM_DATE')
        
    return LAI_df_list,biomass_df_list,LMA_df_list,prod_df_list,planting_df_list,harvest_df_list

def read_RS_product():
    GPP_SLOPE = pd.read_csv('I:/SLOPE_GPP/GPP_extract.csv')
    GPP_SLOPE['Date'] = pd.to_datetime(GPP_SLOPE['Date'],format='%Y-%m-%d')
    LAI_RS = pd.read_csv('I:/LAI_product_ChongyaJiang/LAI_extract.csv')
    LAI_RS['Date'] = pd.to_datetime(LAI_RS['Date'],format='%Y-%m-%d') 
    
    return GPP_SLOPE, LAI_RS

if __name__ == '__main__':
    siteList = ['US-Ne1','US-Ne2','US-Ne3']
    period = [20010101, 20121231]
    
    ## site flux
    NEE_data,GPP_data,Reco_data,energyBalance,ET_data,SWC_data,SWC_data_10cm_AVE,SWC_data_30cm_AVE = read_flux(siteList, period)
        
    n=[0,2500]
    # n=None
    plot_test_series(sList=[NEE_data[t].tolist() for t in siteList],
                     labelList=siteList,n = n,title='NEE',dateCol=[NEE_data['Date'].tolist()]*3)
    plot_test_series(sList=[GPP_data[t].tolist() for t in siteList],
                     labelList=siteList,n = n,title='GPP',dateCol=[GPP_data['Date'].tolist()]*3)
    plot_test_series(sList=[Reco_data[t].tolist() for t in siteList],
                     labelList=siteList,n = n,title='Reco',dateCol=[Reco_data['Date'].tolist()]*3)
    
    plot_test_series(sList=[energyBalance[t].tolist() for t in siteList],labelList=siteList,n = n,
                     title='Closure',dateCol=[energyBalance['Date'].tolist()]*3)
    plot_test_series(sList=[ET_data[t].tolist() for t in siteList],labelList=siteList,n = n,title='ET',
                     dateCol=[ET_data['Date'].tolist()]*3)
    plot_test_series(sList=[SWC_data_10cm_AVE[t].tolist() for t in siteList],labelList=siteList,n = n,
                     title='SWC_data_10cm_AVE',lowLim=0,dateCol=[SWC_data_10cm_AVE['Date'].tolist()]*3)
    plot_test_series(sList=[SWC_data_30cm_AVE[t].tolist() for t in siteList],labelList=siteList,n = n,
                     title='SWC_data_30cm_AVE',lowLim=0,dateCol=[SWC_data_30cm_AVE['Date'].tolist()]*3)
    # plot_test_series(sList=[Rn_data[t].tolist() for t in siteList],labelList=siteList,n = n,title='Rn')
    # plot_test_series(sList=[LE_data[t].tolist() for t in siteList],labelList=siteList,n = n,title='LE')
    # plot_test_series(sList=[H_data[t].tolist() for t in siteList],labelList=siteList,n = n,title='H')
    # plot_test_series(sList=[G_data[t].tolist() for t in siteList],labelList=siteList,n = n,title='G')
    
    LAI_df_list,biomass_df_list,LMA_df_list,prod_df_list,planting_df_list,harvest_df_list = read_BIF_id(siteList)
    GPP_SLOPE, LAI_RS = read_RS_product()
    
    plot_test_series(sList=[GPP_SLOPE[t].tolist() for t in siteList],
                     labelList=siteList,n = n,title='GPP_SLOPE',dateCol=[GPP_SLOPE['Date'].tolist()]*3)
    plot_test_series(sList=[LAI_RS[t].tolist() for t in siteList],
                     labelList=siteList,n = n,title='GPP_SLOPE',dateCol=[LAI_RS['Date'].tolist()]*3)
    
    n=[0,20]
    n=None
    plot_test_series(sList=[np.array(t['LAI_TOT'].tolist()).astype(np.float32) for t in LAI_df_list],
                     labelList=siteList,n = n,title='LAI',dateCol=[t['LAI_DATE'].tolist() for t in LAI_df_list],discrete =True)
    
    plot_test_series(sList=[np.array(t['AG_BIOMASS_CROP'].tolist()).astype(np.float32) for t in biomass_df_list],
                     labelList=siteList,n = n,title='biomass',dateCol=[t['AG_BIOMASS_DATE'].tolist() for t in biomass_df_list],
                     scale = 0.01,discrete =True)
    
    plot_test_series(sList=[np.array(t['LMA'].tolist()).astype(np.float32) for t in LMA_df_list],
                     labelList=siteList,n = n,title='LMA',dateCol=[t['LMA_DATE'].tolist() for t in LMA_df_list],discrete =True)
    
    plot_test_series(sList=[np.array(t['AG_PROD_CROP'].tolist()).astype(np.float32) for t in prod_df_list],
                     labelList=siteList,n = n,title='production',
                     dateCol=[t['AG_PROD_DATE_START'].tolist() for t in prod_df_list],discrete =True)
    
    # import pickle
    # with open('US-NE_LAI.pkl', 'wb') as outp:  # Overwrites any existing file.
    #     pickle.dump(LAI_df_list, outp, pickle.HIGHEST_PROTOCOL)