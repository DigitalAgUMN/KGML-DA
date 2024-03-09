# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 13:15:03 2023

@author: yang8460
"""
import matplotlib.pyplot as plt
from osgeo import gdal
import numpy as np
import pandas as pd
from math import floor
from matplotlib.patches import Rectangle
import KGDA_util as util
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
    year = 2012
    outPath = 'pixel_simulation/20230330'
    leftTop = [667380.86, 1914362.44]
    rightBottom = [677335.75, 1906916.18]
    
    ## load basemap (CDL)
    CDLpath = r'H:\GoogleDrive_umn\My Drive\GEE\CDL_cornBelt_TIGER'
    geoimg = gdal.Open(r'%s/%s/FIPS_%s.tif'%(CDLpath,year,17019))
    gt_forward = geoimg.GetGeoTransform()
    gt_reverse = gdal.InvGeoTransform(gt_forward)
    im_proj = geoimg.GetProjection()

    ## local coordinates of the demo area
    px_leftTop, py_leftTop = gdal.ApplyGeoTransform(gt_reverse, leftTop[0], leftTop[1])
    px_rightBottom, py_rightBottom = gdal.ApplyGeoTransform(gt_reverse, rightBottom[0], rightBottom[1])
    demoArea_w = np.abs(px_leftTop-px_rightBottom)
    demoArea_h = np.abs(py_leftTop-py_rightBottom)
    
    imgCDL = geoimg.ReadAsArray()  
    croplandLoc = ((imgCDL==1)|(imgCDL==5))

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
    
    # save result
    for crop in ['corn','soybean']:
        for mode in ['op','DA']:
            res = yield_map_all[crop][mode]
            write_geo_tiff(im_proj, gt_forward, img=res,
                           path='%s/yield_%s_%s.tif'%(outPath,crop,mode), dataType = 'Float32',NoData=np.nan)
    
    # plot simulated yield    
    fig = plt.figure(constrained_layout=True,figsize=(10,11))
    subfigs = fig.subfigures(2, 1)
    for crop,subfig,d in zip(['corn','soybean'],subfigs.flat,[['Open-loop','DA'],['Open-loop','DA']]):
        yield_map_case = yield_map_all[crop]
        # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13, 8))
        axes = subfig.subplots(nrows=1, ncols=2)
        tmp_map =  np.stack([yield_map_case[mode] for mode in ['op','DA']])
        for mode,ax,descrip in zip(['op','DA'],axes,d):      
            # tmp = ax.imshow(yield_map_case[mode],vmin = 100, vmax = 170, cmap='RdYlGn')
            vmin=np.round(np.nanpercentile(tmp_map,5))
            vmax=np.round(np.nanpercentile(tmp_map,95))
            print('vmin %s, vmax %s'%(vmin, vmax))
            
            tmp = ax.imshow(yield_map_case[mode], vmin=vmin,vmax=vmax, cmap='RdYlGn')
            meanyield = np.nanmean(yield_map_case[mode])
            ax.axis('off')
            # ax.text(0.0, 1.0, s='%s: %s'%(descrip,crop), fontsize=24,transform=ax.transAxes)
            if crop=='corn':
                ax.set_title('%s'%(descrip), fontsize=30)
            # Add a rectangle to the axes
            rect = Rectangle((px_leftTop, py_leftTop), demoArea_w, demoArea_h, edgecolor='k', 
                             facecolor='none', linestyle='--', linewidth=2)
            ax.add_patch(rect)

        # subfig.tight_layout()
        cbar = fig.colorbar(tmp,ax=axes.ravel().tolist(),shrink=0.8)
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label('Grain yield (Bu/Arce)', fontsize=20)
        
    fig.text(-0.06, 0.72, 'Corn', va='center', rotation='vertical',fontsize=30)
    fig.text(-0.06, 0.26, 'Soybean', va='center', rotation='vertical',fontsize=30)
    
    
    for crop,subfig,d in zip(['corn','soybean'],subfigs.flat,[['Open-loop (2012)','DA (2012)'],['Open-loop (2012)','DA (2012)']]):
        yield_map_case = yield_map_all[crop]
        # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13, 8))
        fig,axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 10))
        tmp_map =  np.stack([yield_map_case[mode] for mode in ['op','DA']])
        for mode,ax,descrip in zip(['op','DA'],axes,d):      

            if crop=='corn':
                vmin=118
                vmax=165
            else:
                vmin=36
                vmax=49
                
            print('vmin %s, vmax %s'%(vmin, vmax))
            tmp = ax.imshow(yield_map_case[mode], vmin=vmin,vmax=vmax, cmap='RdYlGn')
            meanyield = np.nanmean(yield_map_case[mode])
            ax.axis('off')

            ax.set_title('%s'%(descrip), fontsize=20)
            # Add a rectangle to the axes
            rect = Rectangle((px_leftTop, py_leftTop), demoArea_w, demoArea_h, edgecolor='k', 
                             facecolor='none', linestyle='--', linewidth=1)
            ax.add_patch(rect)

        fig.tight_layout()
        cbar = plt.colorbar(tmp,ax=axes.ravel().tolist(),shrink=0.35)
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label('Grain yield (Bu/Arce)', fontsize=16)
    
        if crop=='corn':
            fig.text(-0.02, 0.5, 'Corn', va='center', rotation='vertical',fontsize=20)
            # fig.text(-0.02, 0.68, '(a)',fontsize=22)
        else:
            fig.text(-0.02, 0.5, 'Soybean', va='center', rotation='vertical',fontsize=20)
            # fig.text(-0.02, 0.68, '(b)',fontsize=22)