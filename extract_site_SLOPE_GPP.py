# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 22:37:47 2022

@author: yang8460

Extract SLOPE GPP by coordinates
"""

import os,glob
import numpy as np
import pandas as pd
import datetime
import calendar
import pygrib
from tqdm import tqdm
import geopandas as gpd
import time
import matplotlib.pyplot as plt
import shutil
from osgeo import gdal
import struct
from pyproj import Transformer
from typing import List

def get_data_list(time_range: List[str]) -> List[str]:

    file_pattern = '{:4d}/GPP.{:4d}.{:02d}.{:02d}.h10v04.tif' 
    
    file_list = []
    for date in time_range:
        date_obj = pd.to_datetime(date)
        year = date_obj.year
        month = date_obj.month
        day = date_obj.day

        filename = file_pattern.format(year, year, month, day)
        file_list.append(filename)
    return file_list

def extractData(filename,xs,ys,scale=1):
    geoimg = gdal.Open(filename)
    im_proj = geoimg.GetProjection()
    
    gt_forward = geoimg .GetGeoTransform()
    gt_reverse = gdal.InvGeoTransform(gt_forward)
    rb = geoimg.GetRasterBand(1)
    
    #Convert from map to pixel coordinates.
    extracted = []
    for x,y in zip(xs,ys):
        px, py = gdal.ApplyGeoTransform(gt_reverse, x, y)
    
        structval=rb.ReadRaster(px,py,1,1,buf_type=gdal.GDT_UInt16) #Assumes 16 bit int aka 'short'
        intval = struct.unpack('h' , structval)[0]*scale #use the 'short' format code (2 bytes) not int (4 bytes)
        extracted.append(intval)
    return extracted

if __name__ == "__main__":
     
    data_dir = 'GPP250m'
    outName = 'GPP_extrac.csv'
    start_date = '2001-01-01'
    end_date = '2013-01-01'
    scale = 0.001
    
    site_info = pd.DataFrame(columns=['Site_ID','Lon','Lat'])
    site_info['Site_ID'] = ['US-Ne1','US-Ne2','US-Ne3']
    site_info['Lon'] = [-96.4766,-96.4701,-96.4397]
    site_info['Lat'] = [41.1651,41.1649,41.1797]

    # retrive Sinusoidal coordinates
    transformer = Transformer.from_crs("EPSG:4326", 
                                       "+proj=sinu +R=6371007.181 +nadgrids=@null +wktext", always_xy=True)
    tmp = [transformer.transform(lon,lat) for lon,lat in zip(site_info['Lon'],site_info['Lat'])]
    site_info['MODIS_x'] = [t[0] for t in tmp]
    site_info['MODIS_y'] = [t[1] for t in tmp]
    
    # calculate filelist
    time_range = np.arange(start_date, end_date,dtype='datetime64[D]')
    file_list = get_data_list(time_range)
    
    # extract data
    df_extract = pd.DataFrame(columns=['Date']+site_info['Site_ID'].tolist())
    tmp = []
    for i,filename in enumerate(file_list):           
        tmp.append(extractData(filename='%s/%s'%(data_dir,filename),
                               xs=site_info['MODIS_x'].tolist(),ys=site_info['MODIS_y'].tolist(),scale=scale))
        if i%100==0:
            print('%d/%d finished'%(i,len(file_list)))
            
    tmp = np.array(tmp)    
    for i,site in enumerate(site_info['Site_ID'].tolist()):
        df_extract[site] = tmp[:,i]
    df_extract['Date'] = time_range
    
    df_extract.to_csv(outName)
    