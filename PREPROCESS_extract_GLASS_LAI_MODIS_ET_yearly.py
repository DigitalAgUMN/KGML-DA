# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 10:36:24 2022

@author: yang8460
"""

import pandas as pd
import numpy as np
import math
from osgeo import gdal
import struct
from pyproj import Proj
from pyproj import Transformer
import os
from math import floor
import glob
import KGDA_util as util

def get_data_list_GLASS_LAI(time_range,h,v,data_dir):
    file_pattern = 'GLASS01D01.V60.*{:4d}{:03d}.h%2dv%02d.*.hdf'%(h,v) 
    file_list = []
    validTime = []
    for date in time_range:
        date_obj = pd.to_datetime(date)
        year = date_obj.year
        month = date_obj.month
        day = date_obj.day
        doy = date_obj.day_of_year
        filename = file_pattern.format(year, doy)
        path=glob.glob('%s/%s/%s'%(data_dir,year,filename))
        if len(path)>0:
            file_list.append(path[0])
            validTime.append(date_obj)
    return file_list,validTime

def get_data_list_MODIS_ET(time_range,h,v,data_dir):
    file_pattern = 'MOD16A2GF.*{:4d}{:03d}.h%2dv%02d.*.hdf'%(h,v) 
    file_list = []
    validTime = []
    for date in time_range:
        date_obj = pd.to_datetime(date)
        year = date_obj.year
        month = date_obj.month
        day = date_obj.day
        doy = date_obj.day_of_year
        filename = file_pattern.format(year, doy)
        path=glob.glob('%s/%s/%s'%(data_dir,year,filename))
        if len(path)>0:
            file_list.append(path[0])
            validTime.append(date_obj)
    return file_list,validTime

def lat_lon_to_modis(lat, lon):
    
    CELLS = 2400
    VERTICAL_TILES = 18
    HORIZONTAL_TILES = 36
    EARTH_RADIUS = 6371007.181
    EARTH_WIDTH = 2 * math.pi * EARTH_RADIUS
    
    TILE_WIDTH = EARTH_WIDTH / HORIZONTAL_TILES
    TILE_HEIGHT = TILE_WIDTH
    CELL_SIZE = TILE_WIDTH / CELLS
    MODIS_GRID = Proj(f'+proj=sinu +R={EARTH_RADIUS} +nadgrids=@null +wktext')

    x, y = MODIS_GRID(lon, lat)
    h = (EARTH_WIDTH * .5 + x) / TILE_WIDTH
    v = -(EARTH_WIDTH * .25 + y - (VERTICAL_TILES - 0) * TILE_HEIGHT) / TILE_HEIGHT
    return int(h), int(v)

def designateCountyForPoints(siteinfo_outPath = 'E:/syntheticDataset_ecosys/site_info_5133.csv'):
    import geopandas as gpd
        
    ## load sites location
    site_info = pd.DataFrame(columns=['Site_ID','Lon','Lat'])
    stateList = ['Indiana','Illinois','Iowa']
    stateAbbList = ['IN','IL','IA']
    IDList = []
    LonList = []
    LatList = []
    DEMList = []
    pointsGeometry = []

    for state,abb in zip(stateList,stateAbbList):
        site_shp = gpd.read_file('E:/shp/GEE/randomPoints_corn_belt_10000_%s.shp'%state)
        site_dem = pd.read_csv('E:/shp/GEE/pointStats_DEM_%s.csv'%abb)

        IDList.extend(['%s_%s'%(state,t) for t in site_shp.index])
        LonList.extend(site_shp.geometry.x)
        LatList.extend(site_shp.geometry.y)
        pointsGeometry.extend(site_shp.geometry)
        DEMList.extend(site_dem['elevation'].values)
    site_info['Site_ID'] = IDList
    site_info['Lon'] = LonList
    site_info['Lat'] = LatList
    site_info['Alt'] = DEMList    
    
    ## load county shp and classify points by county
    county_shp = gpd.read_file('E:/shp/counties_3I.shp')
    countyList = county_shp.NAME.tolist()
    FIPSList = []
    for point in pointsGeometry:
        info = county_shp[county_shp.contains(point)]
        if len(info)>0:
            FIPSList.append(int(info['FIPS']))
        else:
            FIPSList.append(None)
    
    site_info['FIPS'] = FIPSList
    
    ## retrive Sinusoidal coordinates
    transformer = Transformer.from_crs("EPSG:4326", 
                                       "+proj=sinu +R=6371007.181 +nadgrids=@null +wktext", always_xy=True)
    tmp = [transformer.transform(lon,lat) for lon,lat in zip(site_info['Lon'],site_info['Lat'])]
    site_info['MODIS_x'] = [t[0] for t in tmp]
    site_info['MODIS_y'] = [t[1] for t in tmp]
    
    ## retrive MODIS tile number
    tmp = [lat_lon_to_modis(lat, lon) for lon,lat in zip(site_info['Lon'],site_info['Lat'])]
    site_info['MODIS_h'] = [t[0] for t in tmp]
    site_info['MODIS_v'] = [t[1] for t in tmp]
    
    site_info.to_csv(siteinfo_outPath)    

def makeSiteInfo(pointsFile, DEMfile=None, siteinfo_outPath=None):
        
    ## load sites location
    site_info = pd.DataFrame()
    
    site_basic = pd.read_csv(pointsFile)
    IDList = site_basic['Site'].tolist().copy() 
    site_info['Site_ID'] = IDList
    site_info['Lon'] = site_basic['Lon'].tolist().copy()
    site_info['Lat'] = site_basic['Lat'].tolist().copy()
    site_info['FIPS'] = site_basic['FIPS'].tolist().copy()
    site_info['EPSG5070_x'] = site_basic['x'].tolist().copy()
    site_info['EPSG5070_y'] = site_basic['y'].tolist().copy()
    
    if DEMfile is not None:
        site_dem = pd.read_csv(DEMfile)
        if (IDList == site_dem['Site'].tolist().copy()):
            site_info['Alt'] = site_dem['DEM'].tolist().copy()
        else:
            raise ValueError
    
    ## retrive Sinusoidal coordinates
    transformer = Transformer.from_crs("EPSG:4326", 
                                       "+proj=sinu +R=6371007.181 +nadgrids=@null +wktext", always_xy=True)
    tmp = [transformer.transform(lon,lat) for lon,lat in zip(site_info['Lon'],site_info['Lat'])]
    site_info['MODIS_x'] = [t[0] for t in tmp]
    site_info['MODIS_y'] = [t[1] for t in tmp]
    
    ## retrive MODIS tile number
    tmp = [lat_lon_to_modis(lat, lon) for lon,lat in zip(site_info['Lon'],site_info['Lat'])]
    site_info['MODIS_h'] = [t[0] for t in tmp]
    site_info['MODIS_v'] = [t[1] for t in tmp]
    
    site_info.to_csv(siteinfo_outPath) 
    
def extractData(filename,xs,ys,scale=1,Dtype= 'GLASS_LAI'):
    """

    Parameters
    ----------
    filename : TYPE
        DESCRIPTION.
    xs : list
        x in target CRS.
    ys : list
        y in target CRS.
    scale : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    extracted : list
        extracted vaule.

    """
    if Dtype=='GLASS_LAI':
        geoimg = gdal.Open(filename)
    elif Dtype=='MODIS_ET':
        geoimg = gdal.Open('HDF4_EOS:EOS_GRID:"%s":MOD_Grid_MOD16A2:ET_500m'%filename)
        scale = scale/8 # MODIS ET is 8 day sum
    # SubDatasets = geoimg.GetSubDatasets()
    # Metadata = geoimg.GetMetadata()
    
    im_proj = geoimg.GetProjection()
    
    gt_forward = geoimg.GetGeoTransform()
    gt_reverse = gdal.InvGeoTransform(gt_forward)
    rb = geoimg.GetRasterBand(1)
    
    #Convert from map to pixel coordinates.
    extracted = []
    n = 0
    for x,y in zip(xs,ys):
        px, py = gdal.ApplyGeoTransform(gt_reverse, x, y)
        
        # floor px,py because they are index from zero
        structval=rb.ReadRaster(floor(px),floor(py),1,1,buf_type=gdal.GDT_UInt16) #Assumes 16 bit int aka 'short'
        intval = struct.unpack('h' ,structval)[0]*scale #use the 'short' format code (2 bytes) not int (4 bytes)
        extracted.append(intval)
        n+=1
    return extracted

def extractSINU(siteFile,data_dir,year,Dtype= 'GLASS_LAI'):
    
    site_info = pd.read_csv(siteFile)
    hList = [10,11,12]
    vList = [4,5]

    start_date = '%d-01-01'%year
    end_date = '%d-01-01'%(year+1)
    
    dic_extract = {}
    for h in hList:
        for v in vList:
            if (h==12)&(v==5):
                continue
            print('processing h %s, v %s'%(h,v))
            sites_tile = site_info[(site_info.MODIS_h==h)&(site_info.MODIS_v==v)]
                       
            # calculate filelist
            time_range = np.arange(start_date, end_date, 8, dtype='datetime64[D]')
            if Dtype=='GLASS_LAI':
                file_list,validTime = get_data_list_GLASS_LAI(time_range,h=h,v=v,data_dir=data_dir)
            elif Dtype=='MODIS_ET':
                file_list,validTime = get_data_list_MODIS_ET(time_range,h=h,v=v,data_dir=data_dir)
            tmp = []
            for i,filename in enumerate(file_list):
                tmp.append(extractData(filename=filename,
                                       xs=sites_tile['MODIS_x'].tolist(),
                                       ys=sites_tile['MODIS_y'].tolist(),scale=0.1,Dtype= Dtype))
                # if i%10==0:
                #     print('%d/%d finished for h%2dv%02d'%(i,len(file_list),h,v))
            
            # save data
            tmp = np.array(tmp)        
            for i,site in enumerate(sites_tile['Site_ID'].tolist()):
                dic_extract[site] = tmp[:,i]

    ## ave by county
    points = list(dic_extract.keys())
    FIPS_list = list(set([t[:5] for t in points]))
    FIPS_list.sort()
    
    # create empty dic
    classify_dic = {}
    for FIPS in FIPS_list:
        classify_dic[FIPS] = []
        
    # classify points
    for p in points:
        tmp = dic_extract[p]
        
        # NaN pixel, max LAI (Nan = 255*0.1 = 25.5) and daily ET (Nan = 32762*0.1/8=409.525) can't excess 25
        if max(tmp > 25):
            continue
            
        classify_dic[p[:5]].append(tmp)
        
    # ave
    ave_dic = {}
    for FIPS in FIPS_list:
        ave_dic[FIPS] = np.mean(np.array(classify_dic[FIPS]),axis=0).astype(np.float32)
    
    return ave_dic,validTime

def mkdir(outPath):
    if not os.path.exists(outPath):
        os.makedirs(outPath)

def classifyObs(dic_obs_all, dic_obs_dates,yearSpan,FIPS):
    dataDic = {}
    dataDic['data'] = {}
    dataDic['date'] = {}
    for year in yearSpan:
        dataDic['data'][year] = dic_obs_all[year].get(FIPS)
        dataDic['date'][year] = dic_obs_dates[year]
    return dataDic

def savePkl(dataDic,yearSpan,FIPS,crop,ObsOutPath,item):
    NoneCheck=[dataDic['data'][year] is None for year in yearSpan]
    if min(NoneCheck):
        print('all None for %s %s'%(FIPS,crop))
    else:
        util.save_object(dataDic,'%s/%s_%s_%s.pkl'%(ObsOutPath,item,FIPS,crop))
        
if __name__ == '__main__':
    # a = util.load_object(r'F:\MidWest_counties\MODIS_ET\MODIS_ET_17021_corn.pkl')
    basePath = 'F:/MidWest_counties'
    yearSpan = [t for t in range(2000,2020+1)]
    
    # make yearly site info
    mkdir('%s/site_info_yearly'%basePath)
    for crop in ['corn','soy']:
        for year in yearSpan:
            # load county-level sites and cal MODIS tile number
            pointsFile = '%s/CDL_points_yearly/County_level_samples_merged_TIGER_%s_%s.csv'%(basePath,crop,year)
            siteinfo_outPath = '%s/site_info_yearly/site_info_CDL_%s_%s.csv'%(basePath,crop,year)            
            if not os.path.exists(siteinfo_outPath):
                makeSiteInfo(pointsFile, siteinfo_outPath=siteinfo_outPath)

    # extract MODIS ET
    data_dir = 'E:/MODIS_ET/ET500m'
    dic_ET_all_corn = {}
    dic_ET_all_soy = {}
    dic_ET_dates_corn = {}
    dic_ET_dates_soy = {}
    for crop in ['corn','soy']:
        for year in yearSpan:
            print('processing %s %s'%(crop,year))
            siteinfo_outPath = '%s/site_info_yearly/site_info_CDL_%s_%s.csv'%(basePath,crop,year)   
          
            ave_dic,validTime = extractSINU(siteFile=siteinfo_outPath,data_dir = data_dir, 
                                            year=year,Dtype= 'MODIS_ET')
            if crop=='corn':
                dic_ET_all_corn[year] = ave_dic
                dic_ET_dates_corn[year] = validTime
            else:
                dic_ET_all_soy[year] = ave_dic
                dic_ET_dates_soy[year] = validTime
                
    # save pkl for each county
    ObsOutPath='%s/MODIS_ET_v2'%basePath
    mkdir(ObsOutPath)
    FIPS_list = util.cornBeltCounty()
    print('Saving ET...')
    for FIPS in FIPS_list:
        dataDic_corn = classifyObs(dic_ET_all_corn, dic_ET_dates_corn,yearSpan,FIPS)
        dataDic_soy = classifyObs(dic_ET_all_soy, dic_ET_dates_soy,yearSpan,FIPS)
        
        savePkl(dataDic=dataDic_corn,yearSpan=yearSpan,
                FIPS=FIPS,crop='corn',ObsOutPath=ObsOutPath,item='MODIS_ET')
        savePkl(dataDic=dataDic_soy,yearSpan=yearSpan,
                FIPS=FIPS,crop='soybean',ObsOutPath=ObsOutPath,item='MODIS_ET')
        
    # extract GLASS LAI
    data_dir = 'E:/GLASS_LAI/LAI250m'
    dic_LAI_all_corn = {}
    dic_LAI_all_soy = {}
    dic_LAI_dates_corn = {}
    dic_LAI_dates_soy = {}
    for crop in ['corn','soy']:
        for year in yearSpan:
            print('processing %s %s'%(crop,year))
            siteinfo_outPath = '%s/site_info_yearly/site_info_CDL_%s_%s.csv'%(basePath,crop,year)   
          
            ave_dic,validTime = extractSINU(siteFile=siteinfo_outPath,data_dir = data_dir, 
                                            year=year,Dtype= 'GLASS_LAI')
            if crop=='corn':
                dic_LAI_all_corn[year] = ave_dic
                dic_LAI_dates_corn[year] = validTime
            else:
                dic_LAI_all_soy[year] = ave_dic
                dic_LAI_dates_soy[year] = validTime
                
    # save pkl for each county
    ObsOutPath='%s/GLASS_LAI_v2'%basePath
    mkdir(ObsOutPath)
    FIPS_list = util.cornBeltCounty()
    print('Saving LAI...')
    for FIPS in FIPS_list:
        dataDic_corn = classifyObs(dic_LAI_all_corn, dic_LAI_dates_corn,yearSpan,FIPS)
        dataDic_soy = classifyObs(dic_LAI_all_soy, dic_LAI_dates_soy,yearSpan,FIPS)
        
        savePkl(dataDic=dataDic_corn,yearSpan=yearSpan,
                FIPS=FIPS,crop='corn',ObsOutPath=ObsOutPath,item='GLASS_LAI')
        savePkl(dataDic=dataDic_soy,yearSpan=yearSpan,
                FIPS=FIPS,crop='soybean',ObsOutPath=ObsOutPath,item='GLASS_LAI')
    print('Saving Finished.')
    
