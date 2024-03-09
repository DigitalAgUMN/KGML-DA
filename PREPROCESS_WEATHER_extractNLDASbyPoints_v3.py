# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 10:47:24 2022

@author: yang8460

v2: use the unique index(sampleLoc.csv) to reduce storage and computation
v3: modified the "np2csv" so it will not encounter the out of memory issue when points are too many
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
import sys
version = int(sys.version.split()[0].split('.')[1])
if version > 7:
    import pickle
else:
    import pickle5 as pickle

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
        
def load_object(filename):
    with open(filename, 'rb') as inp:
        data = pickle.load(inp)
    return data
    
def qair2rh(qair, temp, press=1013.25):
    """
    :param qair: specific humidity, dimensionless (e.g. kg/kg) ratio of water mass / total air mass
    :param temp: degrees C
    :param press: pressure in mb
    :return: rh relative humidity, ratio of actual water mixing ratio to saturation mixing ratio
    """
    es = 6.112 * np.exp((17.67 * temp) / (temp + 243.5))
    e = qair * press / (0.378 * qair + 0.622)
    rh = 100 * e / es
    rh[rh > 100] = 100
    rh[rh < 0] = 0
    return rh


def read_NLDAS2(filename):
    # grbs = pygrib.open(filename)
    # grb = grbs[1]
    # lat, lon = grb.latlons()
    grbs = pygrib.index(filename, "parameterName")
    lat, lon = grbs.select(parameterName='205')[0].latlons()
    
    lwdown_tmp = grbs.select(parameterName='205')[0].values  # LW radiation flux downwards (surface) [W/m^2]
    if np.ma.isMaskedArray(lwdown_tmp[:]):  # W/m^2
        lwdown = lwdown_tmp[:].data
        lwdown[lwdown_tmp[:].mask] = np.nan
    else:
        lwdown = lwdown_tmp[:]
    
    precip_tmp = grbs.select(parameterName='61')[0].values  # Precipitation hourly total [kg/m^2]
    if np.ma.isMaskedArray(precip_tmp[:]):
        precip = precip_tmp[:].data
        precip[precip_tmp[:].mask] = np.nan
    else:
        precip = precip_tmp[:]
    
    psurf_tmp = grbs.select(parameterName='1')[0].values  # Surface pressure [Pa]
    if np.ma.isMaskedArray(psurf_tmp[:]):
        psurf = psurf_tmp[:].data
        psurf[psurf_tmp[:].mask] = np.nan
    else:
        psurf = psurf_tmp[:]
    
    spfh_tmp = grbs.select(parameterName='51')[0].values  # 2-m above ground Specific humidity [kg/kg]
    if np.ma.isMaskedArray(spfh_tmp[:]):
        spfh = spfh_tmp[:].data
        spfh[spfh_tmp[:].mask] = np.nan
    else:
        spfh = spfh_tmp[:]
    
    swdown_tmp = grbs.select(parameterName='204')[0].values  # SW radiation flux downwards (surface) [W/m^2]
    if np.ma.isMaskedArray(swdown_tmp[:]):
        swdown = swdown_tmp[:].data
        swdown[swdown_tmp[:].mask] = np.nan
    else:
        swdown = swdown_tmp[:]
    
    tair_tmp = grbs.select(parameterName='11')[0].values  # 2-m above ground Temperature [K]
    if np.ma.isMaskedArray(tair_tmp[:]):
        tair = tair_tmp[:].data
        tair[tair_tmp[:].mask] = np.nan
    else:
        tair = tair_tmp[:]
    
    wind_tmp_Zonal = grbs.select(parameterName='33')[0].values  # 10-m above ground Zonal wind speed [m/s]
    if np.ma.isMaskedArray(wind_tmp_Zonal[:]):
        wind_Zonal = wind_tmp_Zonal[:].data
        wind_Zonal[wind_tmp_Zonal[:].mask] = np.nan
    else:
        wind_Zonal = wind_tmp_Zonal[:]
        wind_Zonal[wind_Zonal < 0] = 0
        
    wind_tmp_Meridional = grbs.select(parameterName='34')[0].values  # 10-m above ground Meridional wind speed [m/s]
    if np.ma.isMaskedArray(wind_tmp_Meridional[:]):
        wind_Meridional = wind_tmp_Meridional[:].data
        wind_Meridional[wind_tmp_Meridional[:].mask] = np.nan
    else:
        wind_Meridional = wind_tmp_Meridional[:]
        wind_Meridional[wind_Meridional < 0] = 0
    
    wind = np.sqrt(wind_Zonal**2+wind_Meridional**2)
    spRH = qair2rh(spfh, tair-273.15, psurf/100.0)
    grbs.close()
    return lon, lat, tair-273.15, precip, swdown, spfh, wind, lwdown, spRH, psurf/100.0


def extractFromNLDAS(task,data_dir,site_info,output_dir,NLDAS_Grid_Space=0.125):
    print(task)
    task = int(task)
    year = int(task / 100)
    month = task % 100
    
    breakPoint = output_dir+'psurf_regrid_{:0>4d}_{:0>4d}_{:0>4d}_{:0>2d}.npy'.\
        format(start_year, end_year, year, month)
    if os.path.exists(breakPoint):
        return
    
    # check data avaiblity added by Qi 
    NLDAS_dir = '%s/GRB/%d/'%(data_dir,year)
    dataN = len(glob.glob(NLDAS_dir + 'NLDAS_FORA0125_H.A{:0>4d}{:0>2d}*.grb'.format\
                (year, month)))
    if dataN > 0:
        print("There are %d files for %s-%s"%(dataN,year,month))

        count = 0
        for mday in range(1, 1 + calendar.monthrange(year, month)[1]):
            print('processing %d day...'%mday)
            DOY = datetime.datetime(year, month, mday).timetuple().tm_yday
            for h,hour in enumerate(range(24)):
                if count==0:
                    print('load hourly file and start to matching sites')
                    now = time.time()
                filename = NLDAS_dir + 'NLDAS_FORA0125_H.A{:0>4d}{:0>2d}{:0>2d}.{:0>2d}00.002.grb'.format\
                    (year, month, mday, hour)
                if not os.path.exists(filename):
                    continue
                if count == 0:
                    lon, lat, tair, precip, swdown, spfh, wind, lwdown, spRH, psurf = read_NLDAS2(filename)
                    ## calculate the data_index for each site   2022-5-17 Qi
                    data_indexList = []
                    for sitei in range(len(site_info.Site_ID)):
                        data_index = np.logical_and(np.abs(lon-site_info.Lon[sitei]) <= NLDAS_Grid_Space/2.0,
                                                    np.abs(lat-site_info.Lat[sitei]) <= NLDAS_Grid_Space/2.0)
                        data_indexList.append(data_index)
                    data_indexALL=np.sum(np.stack(data_indexList),axis=0).astype(bool)
                    lonList = lon[data_indexALL]
                    latList = lat[data_indexALL]

                    tair_regrid = np.zeros((calendar.monthrange(year, month)[1], 24, len(lonList))) * np.nan
                    precip_regrid = np.zeros((calendar.monthrange(year, month)[1], 24, len(lonList))) * np.nan
                    swdown_regrid = np.zeros((calendar.monthrange(year, month)[1], 24, len(lonList))) * np.nan
                    spfh_regrid = np.zeros((calendar.monthrange(year, month)[1], 24, len(lonList))) * np.nan
                    wind_regrid = np.zeros((calendar.monthrange(year, month)[1], 24, len(lonList))) * np.nan
                    lwdown_regrid = np.zeros((calendar.monthrange(year, month)[1], 24, len(lonList))) * np.nan
                    spRH_regrid = np.zeros((calendar.monthrange(year, month)[1], 24, len(lonList))) * np.nan
                    psurf_regrid = np.zeros((calendar.monthrange(year, month)[1], 24, len(lonList))) * np.nan
        
                else:
                    _, _, tair, precip, swdown, spfh, wind, lwdown, spRH, psurf = read_NLDAS2(filename)
                if count==0:
                    print('loading and matching take %.2f s'%(time.time()-now))
                    now = time.time()
               
                if np.sum(data_indexALL) == 0:
                    continue
                tair_regrid[mday-1, hour, :] = tair[data_indexALL]
                precip_regrid[mday-1, hour, :] = precip[data_indexALL]
                swdown_regrid[mday-1, hour, :] = swdown[data_indexALL]
                spfh_regrid[mday-1, hour, :] = spfh[data_indexALL]
                wind_regrid[mday-1, hour, :] = wind[data_indexALL]
                lwdown_regrid[mday-1, hour, :] = lwdown[data_indexALL]
                spRH_regrid[mday-1, hour, :] = spRH[data_indexALL]
                psurf_regrid[mday-1, hour, :] = psurf[data_indexALL]
                
                if count==0:
                    print('assignment take %.4f s for %d site'%(float(time.time()-now),len(site_info.Site_ID)))
                # print('processing hour: %d'%h)
                count+=1
        
        # save the geo-location of samples by Qi
        data_df = pd.DataFrame(columns=['lon','lat'])
        data_df['lon'] = lonList
        data_df['lat'] = latList
        data_df.to_csv(output_dir+'sampleLoc.csv')
        
        # save sample data
        np.save(output_dir+'tair_regrid_{:0>4d}_{:0>4d}_{:0>4d}_{:0>2d}'.format(start_year, end_year, year, month), tair_regrid)
        np.save(output_dir+'precip_regrid_{:0>4d}_{:0>4d}_{:0>4d}_{:0>2d}'.format(start_year, end_year, year, month), precip_regrid)
        np.save(output_dir+'swdown_regrid_{:0>4d}_{:0>4d}_{:0>4d}_{:0>2d}'.format(start_year, end_year, year, month), swdown_regrid)
        np.save(output_dir+'spfh_regrid_{:0>4d}_{:0>4d}_{:0>4d}_{:0>2d}'.format(start_year, end_year, year, month), spfh_regrid)
        np.save(output_dir+'spRH_regrid_{:0>4d}_{:0>4d}_{:0>4d}_{:0>2d}'.format(start_year, end_year, year, month), spRH_regrid)
        np.save(output_dir+'wind_regrid_{:0>4d}_{:0>4d}_{:0>4d}_{:0>2d}'.format(start_year, end_year, year, month), wind_regrid)
        np.save(output_dir+'lwdown_regrid_{:0>4d}_{:0>4d}_{:0>4d}_{:0>2d}'.format(start_year, end_year, year, month), lwdown_regrid)
        np.save(output_dir+'psurf_regrid_{:0>4d}_{:0>4d}_{:0>4d}_{:0>2d}'.format(start_year, end_year, year, month), psurf_regrid)
        
    else:
        print("No data aviable for %s-%s"%(year,month))

def getPkl(term,year,output_pkl_dir):
    return load_object('%s/%s_regrid_%s.pkl'%(output_pkl_dir,term,year))
    
def np2csv(output_dir,site_info,start_year,end_year):
    output_site_dir = os.path.join(output_dir, "site")
    if not os.path.exists(output_site_dir):
        os.makedirs(output_site_dir)
    if not os.path.exists('%s/tmp'%output_site_dir):
        os.makedirs('%s/tmp'%output_site_dir)
    output_pkl_dir = os.path.join(output_dir, "NLDAS_pkl")
    if not os.path.exists(output_pkl_dir):
        os.makedirs(output_pkl_dir)
        
    siteList = site_info['Site_ID'].tolist()
    siteList_lon = site_info['Lon'].tolist()
    siteList_lat = site_info['Lat'].tolist()
    
    # load sample information
    data_df = pd.read_csv(output_dir+'sampleLoc.csv')
    sampleLoc = [(lon,lat) for lon,lat in zip(data_df.lon.tolist(),data_df.lat.tolist())]
    dist = lambda x, y: (x[0]-y[0])**2 + (x[1]-y[1])**2
    
    # calculate pair with identical weather
    siteList_lon = site_info['Lon'].tolist()
    siteList_lat = site_info['Lat'].tolist()
    index_cloestList = []
    for sitei in range(site_info.shape[0]):
        # determin the closest sample    
        xy = (siteList_lon[sitei],siteList_lat[sitei])
        cloest = min(sampleLoc, key=lambda co: dist(co, xy))
        index_cloest = sampleLoc.index(cloest)
        index_cloestList.append(index_cloest)
    validRange = list(set(index_cloestList))
    
    # pair the samples
    pairLocList = []
    tmp = np.array(index_cloestList)
    for v in validRange:
        pairLocList.append(np.where(tmp==v)[0])
            
    print('loading npy data...')
    now = time.time()
    for year in range(start_year, end_year + 1):
        tair_regrid_tmp = []
        precip_regrid_tmp = []
        swdown_regrid_tmp = []
        spfh_regrid_tmp = []
        spRH_regrid_tmp = []
        wind_regrid_tmp = []
        lwdown_regrid_tmp = []
        psurf_regrid_tmp = []
    
        for month in range(1, 13):

            file_T = output_dir + 'tair_regrid_{:0>4d}_{:0>4d}_{:0>4d}_{:0>2d}.npy'.format(start_year, end_year, year, month)
            if os.path.exists(file_T):
                tair_regrid = np.load(file_T).astype(np.float32)
                precip_regrid = np.load(
                    output_dir + 'precip_regrid_{:0>4d}_{:0>4d}_{:0>4d}_{:0>2d}.npy'.format(start_year, end_year, year, month)).astype(np.float32)
                swdown_regrid = np.load(
                    output_dir + 'swdown_regrid_{:0>4d}_{:0>4d}_{:0>4d}_{:0>2d}.npy'.format(start_year, end_year, year, month)).astype(np.float32)
                spfh_regrid = np.load(
                    output_dir + 'spfh_regrid_{:0>4d}_{:0>4d}_{:0>4d}_{:0>2d}.npy'.format(start_year, end_year, year, month)).astype(np.float32)
                spRH_regrid = np.load(
                    output_dir + 'spRH_regrid_{:0>4d}_{:0>4d}_{:0>4d}_{:0>2d}.npy'.format(start_year, end_year, year, month)).astype(np.float32)
                wind_regrid = np.load(
                    output_dir + 'wind_regrid_{:0>4d}_{:0>4d}_{:0>4d}_{:0>2d}.npy'.format(start_year, end_year, year, month)).astype(np.float32)
                lwdown_regrid = np.load(
                    output_dir + 'lwdown_regrid_{:0>4d}_{:0>4d}_{:0>4d}_{:0>2d}.npy'.format(start_year, end_year, year, month)).astype(np.float32)
                psurf_regrid = np.load(
                    output_dir + 'psurf_regrid_{:0>4d}_{:0>4d}_{:0>4d}_{:0>2d}.npy'.format(start_year, end_year, year, month)).astype(np.float32)
                
                tair_regrid_tmp.append(tair_regrid)
                precip_regrid_tmp.append(precip_regrid)
                swdown_regrid_tmp.append(swdown_regrid)
                spfh_regrid_tmp.append(spfh_regrid)
                spRH_regrid_tmp.append(spRH_regrid)
                wind_regrid_tmp.append(wind_regrid)
                lwdown_regrid_tmp.append(lwdown_regrid)
                psurf_regrid_tmp.append(psurf_regrid)
            else:
                print('data is missing for %s-%s'%(year,month))
        
        save_object(tair_regrid_tmp,'%s/tair_regrid_%s.pkl'%(output_pkl_dir,year))
        save_object(precip_regrid_tmp,'%s/precip_regrid_%s.pkl'%(output_pkl_dir,year))
        save_object(swdown_regrid_tmp,'%s/swdown_regrid_%s.pkl'%(output_pkl_dir,year))
        save_object(spfh_regrid_tmp,'%s/spfh_regrid_%s.pkl'%(output_pkl_dir,year))
        save_object(spRH_regrid_tmp,'%s/spRH_regrid_%s.pkl'%(output_pkl_dir,year))
        save_object(wind_regrid_tmp,'%s/wind_regrid_%s.pkl'%(output_pkl_dir,year))
        save_object(lwdown_regrid_tmp,'%s/lwdown_regrid_%s.pkl'%(output_pkl_dir,year))
        save_object(psurf_regrid_tmp,'%s/psurf_regrid_%s.pkl'%(output_pkl_dir,year))
        print('processing year %d'%year)
 
    print('loading take %.4f s'%(float(time.time()-now)))
                           
    # mapping label from all sites to unique sites
    uniqueSiteLoc = [t[0] for t in pairLocList]
    uniqueSite = [siteList[t] for t in uniqueSiteLoc]
    mappingList = [uniqueSite[t] for t in index_cloestList]
    mapping_df = pd.DataFrame()
    mapping_df['Site_ID'] = siteList.copy()
    mapping_df['Mapping_to_unique'] = mappingList
    uniqueSite_df = pd.DataFrame()
    uniqueSite_df['Site_ID'] = uniqueSite.copy()
    mapping_df.to_csv('%s/mapping.csv'%output_site_dir)
    uniqueSite_df.to_csv('%s/uniqueSite.csv'%output_site_dir)
    
    # only save the first site of the pair, Qi 2022-8-2
    for year in range(start_year, end_year + 1):
        tair_regrid = getPkl('tair',year,output_pkl_dir)
        precip_regrid = getPkl('precip',year,output_pkl_dir)
        swdown_regrid = getPkl('swdown',year,output_pkl_dir)
        spfh_regrid = getPkl('spfh',year,output_pkl_dir)
        spRH_regrid = getPkl('spRH',year,output_pkl_dir)
        wind_regrid = getPkl('wind',year,output_pkl_dir)
        lwdown_regrid = getPkl('lwdown',year,output_pkl_dir)
        psurf_regrid = getPkl('psurf',year,output_pkl_dir)
                
        for i,sitei in enumerate(uniqueSiteLoc):
            # determin the closest sample    
            index_cloest = index_cloestList[sitei]
            tmp_list = []
            for month in range(1, 13):
                for mday in range(1, 1 + calendar.monthrange(year, month)[1]):
                    DOY = datetime.datetime(year, month, mday).timetuple().tm_yday
                    for hour in range(24):
                        row = (
                            year, DOY, month, mday, hour,
                            tair_regrid[month-1][mday - 1, hour, index_cloest],
                            precip_regrid[month-1][mday - 1, hour, index_cloest],
                            swdown_regrid[month-1][mday - 1, hour, index_cloest],
                            spfh_regrid[month-1][mday - 1, hour, index_cloest],
                            spRH_regrid[month-1][mday - 1, hour, index_cloest],
                            wind_regrid[month-1][mday - 1, hour, index_cloest],
                            lwdown_regrid[month-1][mday - 1, hour, index_cloest],
                            psurf_regrid[month-1][mday - 1, hour, index_cloest],
                        )
                        tmp_list.append(row)
            tmp_list = np.array(tmp_list).astype(np.float32)
            weather_df = pd.DataFrame(tmp_list,columns=['Year','DOY','Month','Day','Hour','tair','precip','swdown',
                                                        'spfh','spRH','wind','lwdown','psurf'])
            weather_df.insert(loc=0, column='Site_ID', value=siteList[sitei])
            weather_df.to_csv("%s/tmp/%s_%s.csv"%(output_site_dir,siteList[sitei],year),index=False)
            
            if i%500==0:
                print('finish %d/%d site '%(i,len(uniqueSiteLoc)))
        print('Year %s finished'%year)
    
    # merge the yearly site to whole period
    for i,sitei in enumerate(uniqueSiteLoc):
        for year in range(start_year, end_year + 1):
            if year == start_year:
                siteData = pd.read_csv("%s/tmp/%s_%s.csv"%(output_site_dir,siteList[sitei],year))
            else:
                tmp = pd.read_csv("%s/tmp/%s_%s.csv"%(output_site_dir,siteList[sitei],year))
                siteData = pd.concat([siteData, tmp])   
        siteData.to_csv("%s/%s.csv"%(output_site_dir,siteList[sitei]),index=False)
        if i%5==0:
            print('finish %d/%d site '%(i,len(uniqueSiteLoc)))
                
class EcosysClimateData:
    def __init__(self, TTYPE,
                 Z0G = 10, IFLGW = 0, ZNOONG = 12, 
                 PHRG = 7, CN4RIG = 0.25, CNORIG = 0.3, 
                 CPORG = 0.05, CALRG = 0, CFERG = 0, 
                 CCARG =0, CMGRG =0, CNARG =0, CKARG = 0, 
                 CSORG = 0, CCLRG = 0):
        
        self.TTYPE = TTYPE     # TTYPE, Time step, (daily ('D'), hourly ('H'))
        self.CTYPE = 'J'     # CTYPE, Date Format  (Julian (Day of Year) ('J'))
        
        if self.TTYPE =='D':
            self.NI = 2            # Number of Date Variables  (YYYY,DOY)
            self.NN = 6            # Number of Data Variables
            self.IVAR = 'XD'       # Order of Date Variables  (YYYY,DOY)
            self.VAR = 'MNHPWR'    # Order of Data Variables (Max. Temperature (C), Min. Temperature (C), 
                                                           #  Humidity(RH (%)), Precipitation (mm), Wind Speed (m/s),
                                                           #  Radiation daily (MJ/m2/day)))
            self.TYP = 'CCRMSM'         # Units of Weather Variable            
        elif self.TTYPE =='H':
            self.NI = 3           # Number of Date Variables (YYYY,DOY,HOUR)
            self.NN = 5
            self.IVAR = 'XDH'       # Order of Date Variables  (YYYY,DOY,HOUR)
            self.VAR = 'THPWR'         # Order of Data Variables  (Temperature (C), 
                                                           #  Humidity(RH (%)), Precipitation (mm), Wind Speed (m/s),
                                                           #  Radiation Hourly (W/m2))))
            self.TYP = 'CRMSW'         # Units of Weather Variable
            
        self.Z0G = Z0G         # Height of Wind Measurement
        self.IFLGW = IFLGW     # Above the Soil or Vegetation?
        self.ZNOONG = ZNOONG   # Time of solar noon
        self.PHRG = PHRG       # Precipitation Water Quality - pH
        self.CN4RIG = CN4RIG   # Precipitation Water Quality -Ammonium (mg N L-1)
        self.CNORIG = CNORIG   # Precipitation Water Quality -Nitrate (mg N L-1)
        self.CPORG = CPORG     # Precipitation Water Quality - Phosphorus  (mg P L-1)
        self.CALRG = CALRG     # Precipitation Water Quality - Aluminium  (mg Al L-1)
        self.CFERG = CFERG     # Precipitation Water Quality - Iron  (mg Fe L-1)
        self.CCARG = CCARG     # Precipitation Water Quality - Calcium  (mg Ca L-1)
        self.CMGRG = CMGRG     # Precipitation Water Quality - Magnesium  (mg Mg L-1)
        self.CNARG = CNARG     # Precipitation Water Quality - Sodium  (mg Na L-1)
        self.CKARG = CKARG     # Precipitation Water Quality - Potassium  (mg K L-1)
        self.CSORG = CSORG     # Precipitation Water Quality - Sulfate  (mg S L-1)
        self.CCLRG = CCLRG     # Precipitation Water Quality - Chloride  (mg Cl L-1)   

    def write_to_file(self, Climate_Data, Root_Output_Dir):
        # Climate_Data: Radiation (Hourly (W/m2), daily (MJ/m2/day)), 
                       #Temperature (C), Max. Temperature (C), Min. Temperature (C), 
                       #Precipitation (mm), Humidity (RH (%)), Wind Speed (m/s)
        # Climate_Data: pandas data frame, the index is time
                        # The columns are: Temperature (Ta), Max. Temperature (TaMax), Min. Temperature (TaMin)
                        # Precipitation (Prcc), Humidity (RH), Wind Speed (WS), Radiation (SWDW)
        year_range = np.arange(Climate_Data.Year.min(), Climate_Data.Year.max()+1)
        yearList = list(set(Climate_Data.Year))
        
        if not os.path.exists(Root_Output_Dir):
            os.makedirs(Root_Output_Dir)

        for year in year_range:
            if year not in yearList:
                continue
            
            fname = os.path.join(Root_Output_Dir, 'me%0.4d.csv' % year)
            
            tmp_data = Climate_Data[Climate_Data.Year==year]
            
            with open(fname,'w') as f:
                f.write('%s%s%0.2d%0.2d%s%s\n' % (self.TTYPE,self.CTYPE,self.NI,self.NN,self.IVAR,
                                                  self.VAR))
                f.write('%s\n' % self.TYP)
                f.write('%f,%d,%f\n' % (self.Z0G,self.IFLGW,self.ZNOONG))
                f.write((11*('%f,')+'%f\n') % (self.PHRG,self.CN4RIG,self.CNORIG,self.CPORG,
                                            self.CALRG,self.CFERG,self.CCARG,self.CMGRG,
                                            self.CNARG,self.CKARG,self.CSORG,self.CCLRG))
                if self.TTYPE == 'H':
                    
                    for i in range(tmp_data.shape[0]):
                        tmp = tmp_data.iloc[i]
                        DOY = datetime.datetime(int(tmp.Year),int(tmp.Month),int(tmp.Day)).timetuple().tm_yday
                        f.write('%0.4d,%d,%d,%f,%f,%f,%f,%f\n' % (tmp.Year,DOY,tmp.Hour+1,
                                   tmp.Ta, tmp.RH, tmp.Prcc, 
                                   tmp.WS, tmp.SWDW))
                        
                            
                if self.TTYPE == 'D':
                    for DOY in range(1,367):
                        if (not calendar.isleap(year)) and DOY==366:
                            dt = datetime.datetime(year,12,31)
                        else:
                            dt = datetime.datetime(year,1,1) + datetime.timedelta(days=DOY-1)
                        if dt in tmp_data.index:
                            data_hour = tmp_data[tmp_data.index==dt]                        
                            f.write('%0.4d,%d,%f,%f,%f,%f,%f,%f\n' % (year,DOY,data_hour.TaMax.values[0], data_hour.TaMin.values[0],
                                                                data_hour.RH.values[0], data_hour.Prcc.values[0], data_hour.WS.values[0], data_hour.SWDW.values[0]))
                        else:
                            print(dt, ' data is missing')    
            # f.close()

def DewPoint2RH(Ta,Td):
    #
    ea = 0.611*np.exp(17.502*Td/(Td+240.97))
    es = 0.611*np.exp(17.502*Ta/(Ta+240.97))
    return ea*100/es

def write_ecosys_weather(output_dir,site_info, daily=False):
    
    output_site_dir = os.path.join(output_dir, "site")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    siteList = site_info['Site_ID'].tolist()
    
    # load sample information
    data_df = pd.read_csv(output_dir+'sampleLoc.csv')
    sampleLoc = [(lon,lat) for lon,lat in zip(data_df.lon.tolist(),data_df.lat.tolist())]
    dist = lambda x, y: (x[0]-y[0])**2 + (x[1]-y[1])**2
    siteList_lon = site_info['Lon'].tolist()
    siteList_lat = site_info['Lat'].tolist()
    index_cloestList = []
    for sitei in range(site_info.shape[0]):
        # determin the closest sample    
        xy = (siteList_lon[sitei],siteList_lat[sitei])
        cloest = min(sampleLoc, key=lambda co: dist(co, xy))
        index_cloest = sampleLoc.index(cloest)
        index_cloestList.append(index_cloest)
    validRange = list(set(index_cloestList))
    
    # pair the samples
    pairLocList = []
    tmp = np.array(index_cloestList)
    for v in validRange:
        pairLocList.append(np.where(tmp==v)[0])
    
    for pairLoc in tqdm(pairLocList, desc="Prcessing sites"):
        print ('Sample pair are {}'.format([siteList[t] for t in pairLoc]))
        for n, sitei in enumerate(pairLoc):
            # all samples in pairLoc are same
            # run the first one           
            if n==0:
                x1_file = pd.read_csv(os.path.join(output_site_dir, siteList[sitei] + ".csv"))
                # Site_ID,Year,DOY,month,mday,hour,tair,precip,swdown,spfh,spRH,wind,lwdown,psurf
                Climate_Data = pd.DataFrame()
                Climate_Data['Ta'] = x1_file['tair'].values
                Climate_Data['Prcc'] = x1_file['precip'].values
                Climate_Data['RH'] = x1_file['spRH'].values
                Climate_Data['WS'] = x1_file['wind'].values
                Climate_Data['SWDW'] = x1_file['swdown'].values
                Solar_noon = round(12 - siteList_lon[sitei] / 15, 0)
        
                # new method to generate datetime index which is more faster, by Qi 2020-5-19
                # x1_file['connect'] = '-'
                # datetimeList = (x1_file['Year'].astype(str)).str.cat([x1_file['connect'],x1_file['Month'].astype(str),x1_file['connect'],
                #                                         x1_file['Day'].astype(str),x1_file['connect'],x1_file['Hour'].astype(str)])
                # Climate_Data.index = pd.to_datetime(datetimeList,format='%Y-%m-%d-%H')
                Climate_Data['Year'] = x1_file['Year'].values
                Climate_Data['Month'] = x1_file['Month'].values
                Climate_Data['Day'] = x1_file['Day'].values
                Climate_Data['Hour'] = x1_file['Hour'].values
                
                c = EcosysClimateData('H', ZNOONG=Solar_noon)
                # c = EcosysClimateData('H')
                c.write_to_file(Climate_Data, os.path.join(output_site_dir, siteList[sitei], "hourly"))
                # print(Climate_Data.head(5))
                
                if daily:
                    x1_file['connect'] = '-'
                    datetimeList = (x1_file['Year'].astype(str)).str.cat([x1_file['connect'],x1_file['Month'].astype(str),x1_file['connect'],
                                                            x1_file['Day'].astype(str),x1_file['connect'],x1_file['Hour'].astype(str)])
                    Climate_Data.index = pd.to_datetime(datetimeList,format='%Y-%m-%d-%H')
                    
                    climate_data_daily = Climate_Data.resample('D').agg({'Prcc': np.sum, 'RH': np.mean, 'WS': np.mean})
                    climate_data_daily['TaMax'] = Climate_Data['Ta'].resample('D').max()
                    climate_data_daily['TaMin'] = Climate_Data['Ta'].resample('D').min()
                    climate_data_daily['SWDW'] = Climate_Data['SWDW'].resample("D").mean() * 0.0864
                    # print(climate_data_daily.head(5))
            
                    # year,DOY,data_hour.TaMax.values[0], data_hour.TaMin.values[0],
                    # data_hour.RH.values[0], data_hour.Prcc.values[0], data_hour.WS.values[0], data_hour.SWDW.values[0]
                    c = EcosysClimateData('D', ZNOONG=Solar_noon)
                    # c = EcosysClimateData('D')
                    c.write_to_file(climate_data_daily, os.path.join(output_site_dir, siteList[sitei], "daily"))
            else:
                Root_Output_Dir = os.path.join(output_site_dir, siteList[sitei], "hourly")

                template_Dir = os.path.join(output_site_dir,siteList[pairLoc[0]], "hourly")
                
                shutil.copytree(template_Dir,Root_Output_Dir)            
                      
if __name__ == "__main__":
     
    data_dir = 'E:/NLDAS_data/NLDAS_ForcingL4Hourly'
    siteFile = 'F:/MidWest_counties/samplePoints/County_level_samples_merged_TIGER.csv'    
    output_dir = 'F:/MidWest_counties/'
    
    start_year = 1980
    end_year = 2020
    NLDAS_Grid_Space = 0.125
    writeEcosys = False
    genNpy = False
    removeTmp = False
    site_info = pd.read_csv(siteFile)
    site_info.rename(columns={'Site':'Site_ID'},inplace=True)

    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    ## extract data from NLDAS
    dirs = []
    for year in range(start_year,end_year+1):
        for month in range(1,13):
            dirs.append(str('%0.4d%0.2d' % (year, month)))
    
    # save tmp data to npy
    if genNpy:
        for d in dirs:
            extractFromNLDAS(task=d,data_dir=data_dir,site_info=site_info,output_dir=output_dir,NLDAS_Grid_Space=NLDAS_Grid_Space)
    
    # merge npy data to cvs data
    np2csv(output_dir,site_info,start_year,end_year)
    
    ## write ecosys weather files
    if writeEcosys:
        write_ecosys_weather(output_dir,site_info, daily=False)
    
    ## remove tmp file
    if removeTmp:
        tmpfileList = glob.glob('%s/*.npy'%output_dir)
        for t in tmpfileList:
            os.remove(t)