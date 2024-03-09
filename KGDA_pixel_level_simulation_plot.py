# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 14:26:09 2023

@author: yang8460
"""

import matplotlib.pyplot as plt
import os
from osgeo import gdal
import numpy as np
import pandas as pd
import time
import glob
from math import floor

import datetime
import ECONET_util as util
import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.size'] = 12

def write_geo_tiff(im_proj, im_geotrans, img, path, dataType = 'Float32',NoData=-999,BandNames=None):
    
    img[np.isnan(img)] = NoData
    if dataType == 'Float32':
        dType = gdal.GDT_Float32
        img = img.astype(np.float32)
        
    elif dataType == 'Int32':
        dType = gdal.GDT_Int32
        img = img.astype(np.int32)
    
    elif dataType == 'Int16':
        dType = gdal.GDT_Int16
        img = img.astype(np.int16)
    
    elif dataType == 'Uint8':
        dType = gdal.GDT_Byte
        img = img.astype(np.uint8)
        
    else:
        raise ValueError('type error')
        
    # creat geotif
    if len(img.shape) == 2:
        img =img[np.newaxis,:,:]
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, 
                            img.shape[2], img.shape[1], img.shape[0], dType)
    if(dataset!= None):
        dataset.SetGeoTransform(im_geotrans) #写入仿射变换参数
        dataset.SetProjection(im_proj) #写入投影
    for i in range(img.shape[0]):
        band = dataset.GetRasterBand(i+1)
        band.SetNoDataValue(NoData)
        if not BandNames is None:
            band.SetDescription('Band%d_%s'%(i+1,BandNames[i]))
        band.WriteArray(img[i,:,:])
       
    del dataset
    
if __name__ == '__main__':
    year = 2012
    outPath = 'pixel_simulation/20230330'
    
    ## load basemap (CDL)
    CDLpath = r'E:\My Drive\GEE\CDL_cornBelt_TIGER'
    geoimg = gdal.Open(r'%s/%s/FIPS_%s.tif'%(CDLpath,year,17019))
    gt_forward = geoimg.GetGeoTransform()
    gt_reverse = gdal.InvGeoTransform(gt_forward)
    im_proj = geoimg.GetProjection()

    imgCDL = geoimg.ReadAsArray()  
    croplandLoc = ((imgCDL==1)|(imgCDL==5))
    # if crop=='corn':
    #     cropId = 1
    # imgCDL=imgCDL.astype(np.float32)
    # imgCDL[imgCDL!=cropId] = np.nan
    # plt.figure()
    # plt.imshow(imgCDL)
    yield_map_all= {}
    for crop in ['corn','soybean']:
    
        # load results
        resPath = 'pixel_simulation/20230330/res_%s_%s.pkl'%(crop,year)
        data_dic = util.load_object(resPath)
        
        # load points
        points = pd.read_csv(r'F:\MidWest_counties\CDL_points_yearly_Champaign/FIPS_17019_%s_all_pixels_%s.csv'%(crop, year))
        FIDall = list(np.arange(len(points)))
        
        # NASS yield
        NASS_Path = 'F:/MidWest_counties/Yield'
        NASS_yield = util.yieldValidation(NASS_Path)
        if crop == 'corn':
            df_yield_nass = NASS_yield.yield_NASS_corn
            coef = NASS_yield.coef_C2BUacre(0)
        else:
            df_yield_nass = NASS_yield.yield_NASS_soybean
            coef = NASS_yield.coef_C2BUacre(1)
        df_yield_nass.drop(['Unnamed: 0'],inplace=True,axis=1)
        obsYield = df_yield_nass[df_yield_nass['Year']==2012]['17019'].item()
        
        # reverse the point geolocation to array coordinate
        yield_map_case = {}
        for mode in ['op','DA']:
            yield_map = np.zeros([imgCDL.shape[0],imgCDL.shape[1]],dtype=np.float32)*np.nan
            for x,y,FID in zip(points['x'].tolist(),points['y'],FIDall):
                px, py = gdal.ApplyGeoTransform(gt_reverse, x, y)
                value = data_dic[mode][FID][0]
                if value is not None:
                    yield_map[floor(py),floor(px)] = value*coef
                
                if FID % 200000 == 0:
                    print('%s/%s finished'%(FID+1,len(FIDall)))        
            yield_map_case[mode] = yield_map
        
        yield_map_all[crop] = yield_map_case
    
    # for crop in ['corn','soybean']:
    #     yield_map_case = yield_map_all[crop]
    #     fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13, 8))
    #     # axes = subfig.subplots(nrows=1, ncols=2)
    #     tmp_map =  np.stack([yield_map_case[mode] for mode in ['op','DA']])
    #     for mode,ax,descrip in zip(['op','DA'],axes,['(a) Open-loop','(b) DA']):      
    #         # tmp = ax.imshow(yield_map_case[mode],vmin = 100, vmax = 170, cmap='RdYlGn')
            
    #         tmp = ax.imshow(yield_map_case[mode], vmin=np.nanpercentile(tmp_map,5),vmax=np.nanpercentile(tmp_map,95), cmap='RdYlGn')
    #         meanyield = np.nanmean(yield_map_case[mode])
    #         ax.axis('off')
    #         ax.text(0.0, 1.0, s='%s: %s'%(descrip,crop), fontsize=24,transform=ax.transAxes)
    #     plt.tight_layout()
    #     cbar = fig.colorbar(tmp,ax=axes.ravel().tolist(),shrink=0.8)
    #     cbar.ax.tick_params(labelsize=14)
    #     cbar.set_label('Grain yield (Bu/Arce)', fontsize=20)
    
    # save result
    for crop in ['corn','soybean']:
        for mode in ['op','DA']:
            res = yield_map_all[crop][mode]
            write_geo_tiff(im_proj, gt_forward, img=res,
                           path='%s/yield_%s_%s.tif'%(outPath,crop,mode), dataType = 'Float32',NoData=np.nan)
    
    # plot simulated yield    
    fig = plt.figure(constrained_layout=True,figsize=(10,11))
    subfigs = fig.subfigures(2, 1)
    for crop,subfig,d in zip(['corn','soybean'],subfigs.flat,[['(a) Open-loop','(b) DA'],['(c) Open-loop','(d) DA']]):
        yield_map_case = yield_map_all[crop]
        # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13, 8))
        axes = subfig.subplots(nrows=1, ncols=2)
        tmp_map =  np.stack([yield_map_case[mode] for mode in ['op','DA']])
        for mode,ax,descrip in zip(['op','DA'],axes,d):      
            # tmp = ax.imshow(yield_map_case[mode],vmin = 100, vmax = 170, cmap='RdYlGn')
            
            tmp = ax.imshow(yield_map_case[mode], vmin=np.nanpercentile(tmp_map,5),vmax=np.nanpercentile(tmp_map,95), cmap='RdYlGn')
            meanyield = np.nanmean(yield_map_case[mode])
            ax.axis('off')
            ax.text(0.0, 1.0, s='%s: %s'%(descrip,crop), fontsize=24,transform=ax.transAxes)
        # subfig.tight_layout()
        cbar = fig.colorbar(tmp,ax=axes.ravel().tolist(),shrink=0.8)
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label('Grain yield (Bu/Arce)', fontsize=20)
    
    ## show the soil properties
    # load gSSURGO data
    gSSURGO_path = 'I:/gSSURGO_data/county_30m'
    soilRasterDic = {}
    soilProperty_gssurgo = ['BKDS','FC','WP','SCNV','CSAND','CSILT','CORGC']
    soilProperty_ecosys = ['Bulk density','Field capacity','Wilting point','Ks','Sand content','Silt content','SOC']
    for i,t in enumerate(soilProperty_gssurgo):
        geoimg = gdal.Open('%s/FIPS_17019_%s.tif'%(gSSURGO_path,t))
        if i==0:
            gt_forward_s = geoimg.GetGeoTransform()
            gt_reverse_s = gdal.InvGeoTransform(gt_forward_s)
        img = geoimg.ReadAsArray()
        img[img==0] = np.nan
        tmp = np.zeros([imgCDL.shape[0],imgCDL.shape[1]],dtype=np.float32)*np.nan
        tmp[croplandLoc] = img[:,:-1][croplandLoc]
        soilRasterDic[t] = tmp
        
        plt.figure()
        plt.imshow(tmp,vmin=np.nanpercentile(tmp,5),vmax=np.nanpercentile(tmp,95))
        plt.axis('off')
        plt.title(soilProperty_ecosys[i])
   
    ## load the weather files
    # dayl	seconds	0*	86400*	Duration of the daylight period. Based on the period of the day during which the sun is above a hypothetical flat horizon.    
    # prcp	mm	0*	544*	  Daily total precipitation, sum of all forms converted to water-equivalent.    
    # srad	W/m^2	0*	1051*	Incident shortwave radiation flux density, taken as an average over the daylight period of the day.   
    # swe	kg/m^2	0*	13931*	Snow water equivalent, the amount of water contained within the snowpack.   
    # tmax	°C	-60*	60*	Daily maximum 2-meter air temperature. 
    # tmin	°C	-60*	42*	Daily minimum 2-meter air temperature. 
    # vp	Pa	0*	8230*	Daily average partial pressure of water vapor.    
    datetime_series = pd.date_range(start='%s-01-01'%year, end='%s-12-31'%year, freq='D')
    weatherFiles = ['E:/My Drive/GEE DaymetV4_Champaign/%s.tif'%t.strftime('%Y%m%d') for t in datetime_series]
    # discard Dec.31 for leap years
    if len(weatherFiles) > 365:
        weatherFiles = weatherFiles[:-1]
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
    
    # acc prep
    prep = np.nansum(weatherMerge[:,1,:,:],axis=0)
    prep[prep==0] = np.nan
    # plt.figure()
    # plt.imshow(prep)
    # plt.axis('off')
    
    # Tmean
    Tmean = (np.nanmean(weatherMerge[:,4,:,:],axis=0)+np.nanmean(weatherMerge[:,5,:,:],axis=0))/2
    # plt.figure()
    # plt.imshow(Tmean)
    # cbar = plt.colorbar()
    # plt.axis('off')
    # cbar.ax.set_title('Mean T. ℃', fontsize=12)
    # cbar.ax.title.set_position([0.5, 1.05])
    
    # Rad
    rad = np.nanmean(weatherMerge[:,0,:,:],axis=0)*np.nanmean(weatherMerge[:,2,:,:],axis=0)/1e6
    # plt.figure()
    # plt.imshow(rad)
    # cbar = plt.colorbar()
    # plt.axis('off')
    # cbar.ax.set_title('Mean radiation MJ/m2/day', fontsize=12)
    # cbar.ax.title.set_position([0.5, 1.05])
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(11, 8))
    for ax,data,title in zip(axes.flat,[prep,Tmean,rad,soilRasterDic['FC'],soilRasterDic['SCNV'],soilRasterDic['CORGC']],
                             ['Accumulated precipitation mm','Mean temperature ℃','Mean radiation MJ/m2/day',
                              'FC m3/m3','Ks mm/h','SOC gC/kg']):
        tmp = ax.imshow(data, vmin=np.nanpercentile(data,2),vmax=np.nanpercentile(data,98), cmap='viridis') #'summer'
        ax.axis('off')
        cbar = fig.colorbar(tmp,ax=ax,shrink=0.9)
        cbar.ax.tick_params(labelsize=14)
        # cbar.set_label('Grain yield (Bu/Arce)', fontsize=20)
        # ax.text(0.0, -0.1, s=title, fontsize=18,transform=ax.transAxes)
        ax.set_title(title, fontsize=18,y=-0.1)
    plt.tight_layout()
    