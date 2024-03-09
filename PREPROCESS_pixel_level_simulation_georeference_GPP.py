# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 11:35:59 2023

@author: yang8460

The original SLOPE GPP 30m was not geo-referenced!
"""

from osgeo import gdal
from osgeo import osr
import os
import numpy as np
import glob

def write_geo_tiff(img, path, im_geotrans,EPSG=4326,im_proj=None, dataType = 'Float32',NoData=-999,BandNames=None):
    
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
    
    # create crs
    crs = osr.SpatialReference()
    crs.ImportFromEPSG(EPSG)
    
    # creat geotif
    if len(img.shape) == 2:
        img = img[np.newaxis,:,:]
    driver = gdal.GetDriverByName("GTiff")
 
    dataset = driver.Create(path, 
                            img.shape[2], img.shape[1], img.shape[0], dType)
    if(dataset!= None):
        dataset.SetGeoTransform(im_geotrans)
        if im_proj is not None:
           dataset.SetProjection(im_proj)
        else:
           dataset.SetProjection(crs.ExportToWkt())
        
    for i in range(img.shape[0]):
        band = dataset.GetRasterBand(i+1)
        band.SetNoDataValue(NoData)
        if not BandNames is None:
            band.SetDescription('Band%d_%s'%(i+1,BandNames[i]))
        band.WriteArray(img[i,:,:])
    del dataset

def georeferencing(inPath, outPath):
    tmp = inPath.replace('\\','/').split('/')
    imgName = tmp[-1]
    year = tmp[-2]
    geoimg = gdal.Open(inPath)
    im_geotrans = list(geoimg.GetGeoTransform())
    im_proj = geoimg.GetProjection()
    img = geoimg.ReadAsArray()
    
    im_geotrans[0] = 633840
    im_geotrans[3] = 1960530
    write_geo_tiff(img, path='%s/%s/%s'%(outPath,year,imgName),im_geotrans=im_geotrans, EPSG=5070, dataType = 'Int16',NoData=-9999)
    
if __name__ == '__main__':
    path = r'I:/SLOPE_GPP_Champaign_17019/GPP_Daily'
    outPath = r'I:/SLOPE_GPP_Champaign_17019/GPP_Daily_georeferenced'  
    for year in range(2000,2020+1):
        if not os.path.exists('%s/%s'%(outPath,year)):
            os.makedirs('%s/%s'%(outPath,year))
        
        dataList = glob.glob("%s/%s/*.tif"%(path,year))
        for t,inPath in enumerate(dataList):
            georeferencing(inPath, outPath)
            if t%50 == 0:
                print('Processing data of %s, DOY:%s'%(year,t+1))
    