# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 11:50:03 2023

@author: yang8460
"""

import os
import numpy as np
import pandas as pd
from osgeo import gdal
from pyproj import Transformer
import geopandas as gpd
import copy

def mkdir(outPath):
    if not os.path.exists(outPath):
        os.makedirs(outPath)

def coordinatesTransfer(validPoints,gt_forward, resolution = 30):
    
    
    # transform coordinates
    geoxList = []
    geoyList = []
    
    points_df = pd.DataFrame()    
    for row,col in zip(validPoints[0],validPoints[1]):
        px, py = gdal.ApplyGeoTransform(gt_forward, float(col), float(row))  # the col and row is reversed
        
        # re-center the point, if not, the retrieved coordinates will at the lefttop corner
        geoxList.append(px+resolution/2)
        geoyList.append(py-resolution/2)
    points_df['x'] = geoxList
    points_df['y'] = geoyList
    
    # cal Lon Lat
    transformer = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)
    tmp = [transformer.transform(x,y) for x,y in zip(points_df['x'] ,points_df['y'] )]
    points_df['Lon'] = [t[0] for t in tmp]
    points_df['Lat'] = [t[1] for t in tmp]        
    points_gdf = gpd.GeoDataFrame(copy.deepcopy(points_df), crs="EPSG:5070", geometry=gpd.points_from_xy(points_df.x, points_df.y))
    return points_df, points_gdf

def CDLtoPoints(path, year, FIPS, outPath):
    print('Processing %s'%year)
    geoimg = gdal.Open(r'%s/%s/FIPS_%s.tif'%(path,year,FIPS))
    gt_forward = geoimg.GetGeoTransform()
    img = geoimg.ReadAsArray()
    
    # corn & soybean pixels
    loc_corn = np.where(img==1)
    loc_soy = np.where(img==5)
    
    print('Processing corn...')
    # corn
    points_df, points_gdf = coordinatesTransfer(validPoints=loc_corn,gt_forward=gt_forward, resolution = 30)
    points_df.to_csv('%s/FIPS_%s_corn_all_pixels_%s.csv'%(outPath,FIPS,year))
    points_gdf.to_file('%s/FIPS_%s_corn_all_pixels_%s.shp'%(outPath,FIPS,year))
    
    print('Processing soybean...')
    # soybean
    points_df, points_gdf = coordinatesTransfer(validPoints=loc_soy,gt_forward=gt_forward, resolution = 30)
    points_df.to_csv('%s/FIPS_%s_soybean_all_pixels_%s.csv'%(outPath,FIPS,year))
    points_gdf.to_file('%s/FIPS_%s_soybean_all_pixels_%s.shp'%(outPath,FIPS,year))
    
if __name__ == '__main__':
    # path
    CDLpath = r'E:\My Drive\GEE\CDL_cornBelt_TIGER'
    outPath = 'F:/MidWest_counties/CDL_points_yearly_Champaign'
    yearSpan = [t for t in range(2000,2020+1)]
    FIPS = '17019'
    mkdir(outPath)
    for year in yearSpan:
        print('Processing year %s'%year)
        CDLtoPoints(path=CDLpath, year=year, FIPS=FIPS, outPath=outPath)