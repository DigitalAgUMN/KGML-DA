# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 11:27:10 2022

@author: Qi Yang

"""

import SOIL_util as util
from osgeo import gdal
import geopandas as gpd
from math import floor
import struct
import os
import time
import numpy as np

def extractFromgSSURGO(site_shp,gSSURGO_raster_path):
    #the extraction function 
    geoimg = gdal.Open(gSSURGO_raster_path)
    gt_forward = geoimg.GetGeoTransform()
    gt_reverse = gdal.InvGeoTransform(gt_forward)
    rb = geoimg.GetRasterBand(1)
    
    xs = site_shp.x.tolist()
    ys = site_shp.y.tolist()
    #Convert from map to pixel coordinates.
    extracted = []
    n = 0
    scale = 1
    for x,y in zip(xs,ys):
        
        px, py = gdal.ApplyGeoTransform(gt_reverse, x, y)
        
        # floor px,py because they are index from zero
        structval=rb.ReadRaster(floor(px),floor(py),1,1,buf_type=gdal.GDT_UInt32) #Assumes 16 bit int aka 'short'
        intval = struct.unpack('I' ,structval)[0]*scale #use the 'short' format code (2 bytes) not int (4 bytes)
        extracted.append(intval)
        n+=1
    return extracted

def write_ecosys_soil_file(MUKEY_list,abb,Output_location,gSSURGO_location_state,Type,layers_depth=None,siteID=None):
    start = time.time()
    # extracted file of the extraction function
    util.dst_MUKEY(MUKEY_list, abb, Type, gSSURGO_location_state, Output_location,layers_depth=layers_depth,siteID=siteID)
    elapsed_time = time.time() - start
    print("dst takes {:.1f} sec".format(elapsed_time))
    
if __name__ == '__main__':
    
    # outline file's categories
    Type = 'point' 
    
    # site information 
    layers_depth = np.array([0.01,0.05,0.15,0.3,0.6,1.0,2.0])
    stateAbbList = ['ND','SD','NE','KS',
                    'MN','IA','MO','AR',
                    'WI','IL','TN',
                    'MI','IN','KY','OH']
    FIPS_state = ['38','46','31','20',
                  '27','19','29','05',
                  '55','17','47',
                  '26','18','21','39']

    useCONUSraster = True
    
    output_dir = 'F:/MidWest_counties/'
    site_shp = gpd.read_file('F:/MidWest_counties/samplePoints/County_level_samples_merged_TIGER.shp')
    tmp = site_shp['FIPS'].tolist()
    site_shp['FIPS_state'] = [t[:2] for t in tmp]
        
    # parameters
    gSSURGO_raster_path_CONUS = r'E:\gSSURGO_data\October 2021 gSSURGO CONUS\MapunitRaster_30m.tif'   
    gSSURGO_raster_path = r'E:\gSSURGO_data\October 2021 gSSURGO by State\gSSURGO_{}\MapunitRaster_10m.tif'      
    gSSURGO_location_state = r"E:\gSSURGO_data\October 2021 gSSURGO by State\gSSURGO_{}\gSSURGO_{}.gdb"
    Output_location=r'%s/gSSURGO_Ecosys'%output_dir
    
    if not os.path.exists(Output_location):
        os.makedirs(Output_location)

    ## extract data
    print('Extract MUKEY...')
    MUKEY_list_all = []
    for abb,FI in zip(stateAbbList,FIPS_state):
        state_points = site_shp[site_shp.FIPS_state==FI]
        if useCONUSraster:
            MUKEY_list = extractFromgSSURGO(state_points,gSSURGO_raster_path=gSSURGO_raster_path_CONUS)
        else:
            MUKEY_list = extractFromgSSURGO(state_points,gSSURGO_raster_path=gSSURGO_raster_path.format(abb))
        MUKEY_list_all.append(MUKEY_list)
        
        ## write ecosys soil input file
        print('Retrieve properties by MUKEY...')
        write_ecosys_soil_file(MUKEY_list=MUKEY_list,abb =abb, Output_location=Output_location,
                               gSSURGO_location_state=gSSURGO_location_state,Type=Type,
                               layers_depth=layers_depth,siteID=state_points['Site'].tolist())   
      
   

    

