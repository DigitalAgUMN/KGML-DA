# -*- coding: utf-8 -*-
"""
Created on Thu May 12 18:55:29 2022

@author: yang8460
"""

import numpy as np
import os
import pandas as pd
import requests
import datetime

class NASAPowerWeatherDataProvider():
    '''
    resolution: 0.5 degree * 0.5 degree
    unit: https://power.larc.nasa.gov/#resources
    '''
    def __init__(self, latitude, longitude, update=False):
        self.latitude=latitude
        self.longitude=longitude
        self.power_variables = ["TOA_SW_DWN", "ALLSKY_SFC_SW_DWN", "T2M", "T2M_MIN",
                           "T2M_MAX", "T2MDEW", "WS2M", "PRECTOTCORR"]
        self.tmpPath = 'siteData'
        self.HTTP_OK = 200
        if not os.path.exists(self.tmpPath):
            os.mkdir(self.tmpPath)
        self.cache = '%s/weather_lon%.1f_lat%.1f.csv'%(self.tmpPath,self.longitude,self.latitude)
        if update:
            self.query_NASAPower_server()
        else:
            if not os.path.exists(self.cache):
                self.query_NASAPower_server()
            else:
                self.df_power = pd.read_csv(self.cache,index_col=0)
                self.df_power["DAY"] = pd.to_datetime(self.df_power.index, format="%Y%m%d")
        
    def query_NASAPower_server(self):
        """Query the NASA Power server for data on given latitude/longitude
        """  
        start_date = datetime.date(1983,7,1)
        end_date = datetime.date.today()
        
        # build URL for retrieving data, using new NASA POWER api
        server = "https://power.larc.nasa.gov/api/temporal/daily/point"
        payload = {"request": "execute",
                   "parameters": ",".join(self.power_variables),
                   "latitude": self.latitude,
                   "longitude": self.longitude,
                   "start": start_date.strftime("%Y%m%d"),
                   "end": end_date.strftime("%Y%m%d"),
                   "community": "AG",
                   "format": "JSON",
                   "user": "anonymous"
                   }
        msg = "Starting retrieval from NASA Power"   
        print(msg)
        req = requests.get(server, params=payload)
        if req.status_code != self.HTTP_OK:
            msg = ("Failed retrieving POWER data, server returned HTTP " +
                   "code: %i on following URL %s") % (req.status_code, req.url)
            raise ValueError(msg)
        msg = "Successfully retrieved data from NASA Power"
        print(msg)
        powerdata = req.json()     
        self.df_power = self.process_POWER_records(powerdata)
        
        # cal RH
        rh = self.dew2rh(tdew=np.array(self.df_power['T2MDEW']), 
                         tmax=np.array(self.df_power['T2M_MAX']),
                         tmin=np.array(self.df_power['T2M_MIN']))
        self.df_power['RH'] = rh
        
        # remove small rain
        index = self.df_power['PRECTOTCORR'][self.df_power['PRECTOTCORR']<1].index
        self.df_power.loc[index,'PRECTOTCORR'] = 0
        
        # save cache
        self.df_power.to_csv(self.cache)
        
    def dew2rh(self, tdew, tmax, tmin):
        """
        :param dew: dew T
        :param temp: degrees C
        :return: rh relative humidity, ratio of actual water mixing ratio to saturation mixing ratio
        """
        e_0_max = 0.6108 * np.exp(17.27 * tmax / (tmax + 237.3))
        e_0_min = 0.6108 * np.exp(17.27 * tmin / (tmin + 237.3))
        es = (e_0_min + e_0_max) / 2 
        
        ea = 0.6108 * np.exp(17.27 * tdew / (tdew + 237.3))
        rh = 100 * ea / es
        rh[rh > 100] = 100
        rh[rh < 0] = 0
        return rh
    
    def process_POWER_records(self,powerdata):
        """Process the meteorological records returned by NASA POWER
        """
    
        fill_value = float(powerdata["header"]["fill_value"])
    
        df_power = {}
        for varname in self.power_variables:
            s = pd.Series(powerdata["properties"]["parameter"][varname])
            s[s == fill_value] = np.NaN
            df_power[varname] = s
        df_power = pd.DataFrame(df_power)
        df_power["DAY"] = pd.to_datetime(df_power.index, format="%Y%m%d")
    
        # find all rows with one or more missing values (NaN)
        ix = df_power.isnull().any(axis=1)
        # Get all rows without missing values
        df_power = df_power[~ix]
    
        return df_power
    
    def get_daily(self,date = datetime.datetime(2000,1,1)):
        index = self.df_power[self.df_power['DAY']==date].index
        return self.df_power.loc[index]
    
    def get_period(self,period):
        index_s = self.df_power[self.df_power['DAY']==period[0]].index
        index_e = self.df_power[self.df_power['DAY']==period[1]].index
        return self.df_power.loc[index_s[0]:index_e[0]]
    
    def get_monthly(self,year,month):
        
        monthList = [t.month for t in self.df_power['DAY']]
        index = np.array(monthList)==month
        self.df_power['Year'] = [t.year for t in self.df_power['DAY']]
        self.df_power['Month'] = [t.month for t in self.df_power['DAY']]
        index = self.df_power[(self.df_power['Year']==year)&(self.df_power['Month']==month)].index   
        return self.df_power.loc[index]