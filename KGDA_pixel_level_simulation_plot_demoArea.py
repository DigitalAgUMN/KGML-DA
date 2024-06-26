# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 15:16:38 2023

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
import copy
import datetime
import ECONET_util as util
import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['figure.dpi'] = 500
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
    # boundary of the demo area
    # leftTop = [662859.26, 1944224.74]
    # rightBottom = [675679.29, 1934393.69]
    # v2
    leftTop = [667380.86, 1914362.44]
    rightBottom = [677335.75, 1906916.18]
    
    # outPath = 'pixel_simulation/20230331_1200'
    outPath = 'pixel_simulation/20230403_0110_globalPara'
    CDLpath = r'E:\My Drive\GEE\CDL_cornBelt_TIGER'
    
    yield_map_all= {}
    
    for year in [2010,2011,2012,2013,2014]:
          
        ## load basemap (CDL)      
        geoimg = gdal.Open(r'%s/%s/FIPS_%s.tif'%(CDLpath,year,17019))
        gt_forward = geoimg.GetGeoTransform()
        gt_reverse = gdal.InvGeoTransform(gt_forward)
        im_proj = geoimg.GetProjection()
        gt_forward_demo = copy.deepcopy(list(gt_forward))
        gt_forward_demo[0] = leftTop[0]
        gt_forward_demo[3] = leftTop[1]
        
        ## local coordinates of the demo area
        px_leftTop, py_leftTop = gdal.ApplyGeoTransform(gt_reverse, leftTop[0], leftTop[1])
        px_rightBottom, py_rightBottom = gdal.ApplyGeoTransform(gt_reverse, rightBottom[0], rightBottom[1])
        imgCDL = geoimg.ReadAsArray()  
        croplandLoc = ((imgCDL==1)|(imgCDL==5))
        imgCDL_demo = imgCDL[floor(py_leftTop):floor(py_rightBottom),
                             floor(px_leftTop):floor(px_rightBottom)]
        # if crop=='corn':
        #     cropId = 1
        # imgCDL=imgCDL.astype(np.float32)
        # imgCDL[imgCDL!=cropId] = np.nan
        # plt.figure()
        # plt.imshow(imgCDL_demo)
        
        yield_map_year= {}
        for crop in ['corn','soybean']:
        
            # load results
            resPath = '%s/res_demoArea_%s_%s.pkl'%(outPath,crop,year)
            data_dic = util.load_object(resPath)
            
            # load points
            points = pd.read_csv(r'F:\MidWest_counties\CDL_points_yearly_Champaign/FIPS_17019_%s_all_pixels_%s.csv'%(crop, year))
            points = points[(rightBottom[0]>points['x'])&(leftTop[0]<points['x'])&(rightBottom[1]<points['y'])&(leftTop[1]>points['y'])]
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
                    
                    if FID % 20000 == 0:
                        print('%s/%s finished'%(FID+1,len(FIDall)))        
                yield_map_case[mode] = yield_map[floor(py_leftTop):floor(py_rightBottom),
                                     floor(px_leftTop):floor(px_rightBottom)]
            
            yield_map_year[crop] = yield_map_case
        yield_map_all[year] = yield_map_year
    
    
    
       
    for crop in ['corn','soybean']:
        tmp_map =  np.stack([k for i,j in yield_map_all.items() for _,k in j[crop].items()])
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(13, 3.8))
       
        yield_map_case = yield_map_all[year][crop]
        
        # axes = subfig.subplots(nrows=1, ncols=2)
        
        for ax,year in zip(axes[0,:],[2010,2011,2012,2013,2014]):      
            yield_map_case = yield_map_all[year][crop]
            mode = 'op'
            tmp = ax.imshow(yield_map_case[mode], vmin=np.nanpercentile(tmp_map,1),vmax=np.nanpercentile(tmp_map,99), cmap='RdYlGn')
            meanyield = np.nanmean(yield_map_case[mode])
            ax.axis('off')
            # ax.text(0.0, 1.0, s='%s'%(year), fontsize=24,transform=ax.transAxes)
            ax.set_title('%s'%(year), fontsize=18)
        for ax,year in zip(axes[1,:],[2010,2011,2012,2013,2014]):         
            yield_map_case = yield_map_all[year][crop]
            mode = 'DA'
            tmp = ax.imshow(yield_map_case[mode], vmin=np.nanpercentile(tmp_map,1),vmax=np.nanpercentile(tmp_map,99), cmap='RdYlGn')
            meanyield = np.nanmean(yield_map_case[mode])
            ax.axis('off')
            # ax.text(0.0, 1.0, s='%s'%(year), fontsize=24,transform=ax.transAxes)
            # ax.set_title('%s'%(year), fontsize=18)
        plt.tight_layout()
        cbar = fig.colorbar(tmp,ax=axes.ravel().tolist(),shrink=0.8)
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label('Grain yield (Bu/Arce)', fontsize=20)
    
        fig.text(-0.02, 0.69, 'Open-loop', va='center', rotation='vertical',fontsize=22)
        fig.text(-0.02, 0.26, 'DA', va='center', rotation='vertical',fontsize=22)
        
    # save result
    for year in [2010,2011,2012,2013,2014]:
        for crop in ['corn','soybean']:
            for mode in ['op','DA']:
                res = yield_map_all[year][crop][mode]
                write_geo_tiff(im_proj, gt_forward_demo, img=res,
                                path='%s/yield_%s_%s_%s_demoArea.tif'%(outPath,crop,mode,year), dataType = 'Float32',NoData=np.nan)
    
    # # plot simulated yield    
    # fig = plt.figure(constrained_layout=True,figsize=(10,11))
    # subfigs = fig.subfigures(2, 1)
    # for crop,subfig,d in zip(['corn','soybean'],subfigs.flat,[['(a) Open-loop','(b) DA'],['(c) Open-loop','(d) DA']]):
    #     yield_map_case = yield_map_all[crop]
    #     # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13, 8))
    #     axes = subfig.subplots(nrows=1, ncols=2)
    #     tmp_map =  np.stack([yield_map_case[mode] for mode in ['op','DA']])
    #     for mode,ax,descrip in zip(['op','DA'],axes,d):      
    #         # tmp = ax.imshow(yield_map_case[mode],vmin = 100, vmax = 170, cmap='RdYlGn')
            
    #         tmp = ax.imshow(yield_map_case[mode], vmin=np.nanpercentile(tmp_map,5),vmax=np.nanpercentile(tmp_map,95), cmap='RdYlGn')
    #         meanyield = np.nanmean(yield_map_case[mode])
    #         ax.axis('off')
    #         ax.text(0.0, 1.0, s='%s: %s'%(descrip,crop), fontsize=24,transform=ax.transAxes)
    #     # subfig.tight_layout()
    #     cbar = fig.colorbar(tmp,ax=axes.ravel().tolist(),shrink=0.8)
    #     cbar.ax.tick_params(labelsize=14)
    #     cbar.set_label('Grain yield (Bu/Arce)', fontsize=20)
    
   