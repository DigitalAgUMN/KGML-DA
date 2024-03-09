# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 10:57:07 2023

@author: yang8460
"""
from ECONET_NASApower import NASAPowerWeatherDataProvider
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
import scipy.signal
import ECONET_Networks as net
import torch
import math
import ECONET_util as util

# Champaign county location for NASA POWER data
Lon = -88.2
Lat = 40.2
wdp = NASAPowerWeatherDataProvider(latitude=Lat, longitude=Lon, update=False) # a=wdp.df_power
df_power = wdp.df_power
yearList = [2010,2011,2012,2013,2014]

prepList = []
timeline = []
prepAcc = []
for year in yearList:
    tmp = 0
    for m in range(6,10):      
        nasa = wdp.get_monthly(year,m)
        prepList.append(np.sum(nasa['PRECTOTCORR']))
        timeline.append(datetime.datetime(year,m,1))
        tmp += prepList[-1]
    prepAcc.append(np.sum(tmp))
plt.figure(figsize=(13,3))
plt.bar(timeline,prepList)

