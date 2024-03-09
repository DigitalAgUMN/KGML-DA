# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 11:07:58 2023

@author: yang8460
"""

import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.plot import show
import matplotlib.pyplot as plt
import os
from osgeo import gdal
import numpy as np
import pandas as pd
import time

class Soil():
    def __init__(self,soil_layers = 12,layers_depth = np.array([0.01,0.05,0.1,0.15,0.18,0.28,0.35,0.59,0.92,1.32,1.6,2.52])):
        self.S_Column_Parameters = ['CDPTH','BKDS','FC','WP','SCNV','SCNH','CSAND','CSILT','FHOL','ROCK','PH','CEC','AEC','CORGC','CORGR','CORGN',\
                               'CORGP','CNH4','CNO3','CPO4','CAL','CFE','CCA','CMG','CNA','CKA','CSO4','CCL','CALPO','CFEPO','CCAPD',\
                               'CCAPH','CALOH','CFEOH','CCACO','CCASO','GKC4','GKCH','GKCA','GKCM','GKCN','GKCK','THW','THI',\
                               'RSC_1','RSN_1','RSP_1','RSC_0','RSN_0','RSP_0','RSC_2','RSN_2','RSP_2']        
        self.PSIFC = -0.033	#Soil water potential at field capacity
        self.PSIWP = -1.5	#Soil water potential at wilting point
        self.ALBS = 0.2  	#Soil albedo when wet
        self.PH_0 = 5.33   	#Residue pH (-)   # set default as 5.33
        self.RSC_1_0 = 400	#Initial fine plant residue C
        self.RSN_1_0 = 10 	#Initial fine plant residue N
        self.RSP_1_0 = 1  	#Initial fine plant residue P
        self.RSC_0_0 = 0	        #Initial coarse woody residue C
        self.RSN_0_0 = 0  	#Initial coarse woody residue N
        self.RSP_0_0 = 0   	#Initial coarse woody residue P
        self.RSC_2_0 = 0  	#Initial animal manure C
        self.RSN_2_0 = 0  	#Initial animal manure N
        self.RSP_2_0 = 0  	#Initial animal manure P
        self.IXTYP_1 = 1 	#Plant residue type# MAIZE 1; WHEAT 2; SOYBEAN 3; NEW STRAW 4; OLD STRAW 5; COMPOST 6; GREEN MANURE 7; NEW DECIDUOUS FOREST 8; NEW CONIFEROUS FOREST 9; OLD DECIDUOUS FOREST 10; OLD CONIFEROUS FOREST 11
        self.IXTYP_2 = 1	#Animal manure type#  RUMINANT 1; NON-RUMINANT 0
        self.NU = 1	#Soil Surface Layer No.
        self.NJ = soil_layers	 #Additional Layers Below Max. Root Depth 
        self.NL1 = 0	#additional layers below the rooting zone that are specified in the input soil file
        self.NL2 = 1	#additional layers below the rooting zone that are added below the lowest layer indicated by (1)
        self.ISOILR = 0 	#Layer No. at Max. Root Depth
        self.CDPTH = layers_depth	#Depth
        self.BKDS = np.nan*np.ones((soil_layers,), dtype=int)	#Bulk density  (Mg m-3)
        self.FC  = -1*np.ones((soil_layers,), dtype=int)	#Water content at field capacity (m3 m-3)
        self.WP = -1*np.ones((soil_layers,), dtype=int)	#Water content at wilting point (m3 m-3)
        self.SCNV = -1*np.ones((soil_layers,), dtype=int)	#Saturated hydraulic conductivity in the vertical direction (mm h-1)
        self.SCNH = -1*np.ones((soil_layers,), dtype=int)	#Saturated hydraulic conductivity in the horizontal direction (mm h-1)
        self.CSAND = np.nan*np.ones((soil_layers,), dtype=int)	#Sand content  (g kg-1)
        self.CSILT = np.nan*np.ones((soil_layers,), dtype=int)	#Silt content  (g kg-1)
        self.FHOL  = np.zeros((soil_layers,), dtype=int)*1.0	#Volume fraction occupied by macropores    # set default if depath less than 0.3m then to be 0.01, otherwise set as 0
        self.FHOL[self.CDPTH<=0.3]=0.01
        self.ROCK  = 0*np.ones((soil_layers,), dtype=int)	#Volume fraction occupied by coarse fragments  # set default as 0.0
        self.PH = 5.33*np.ones((soil_layers,), dtype=int)	#Solution Chemistry  - pH   # set default as 5.33
        self.CEC = -1*np.ones((soil_layers,), dtype=int)	#Solid Chemistry Cation exchange capacity (cmol+ kg-1)
        self.AEC = 2.5*np.ones((soil_layers,), dtype=int)	#Solid Chemistry Anion exchange capacity  (cmol- kg-1)
        self.CORGC = np.nan*np.ones((soil_layers,), dtype=int)	#Biology Total organic C (g C kg-1)
        self.CORGR = -1*np.ones((soil_layers,), dtype=int)	#Biology articulate organic C  (g C kg-1) 
        self.CORGN = -1*np.ones((soil_layers,), dtype=int)	#Biology organic N  (g N Mg-1)
        self.CORGP = -1*np.ones((soil_layers,), dtype=int)	#Biology Organic P  (g P Mg-1)
        self.CNH4 = 3*np.ones((soil_layers,), dtype=int)	#Solution Chemistry Soluble + exchangeable NH4+ (g N Mg-1)
        self.CNO3 = 12*np.ones((soil_layers,), dtype=int)	#Solution Chemistry  NO3- (g N Mg-1)
        self.CPO4 = 10*np.ones((soil_layers,), dtype=int)	#Solution Chemistry Exchangeable P (g P Mg-1)
        self.CAL = -1*np.ones((soil_layers,), dtype=int)	#Solution Chemistry - Aluminum (g Al Mg-1)
        self.CFE = -1*np.ones((soil_layers,), dtype=int)	#Solution Chemistry Iron  (g Fe Mg-1)
        self.CCA = 100*np.ones((soil_layers,), dtype=int)	#Solution Chemistry Calcium (g Ca Mg-1)
        self.CMG = 120*np.ones((soil_layers,), dtype=int)	#Solution Chemistry Magnesium (g Mg Mg-1)
        self.CNA = 23*np.ones((soil_layers,), dtype=int)	#Solution Chemistry Sodium (g Na Mg-1)
        self.CKA = 39*np.ones((soil_layers,), dtype=int)	#Solution Chemistry Potassium (g K Mg-1)
        self.CSO4 = 48*np.ones((soil_layers,), dtype=int)	#Solution Chemistry Sulfate (g S Mg-1)
        self.CCL = 35*np.ones((soil_layers,), dtype=int)	#Solution Chemistry Chloride (g Cl Mg-1).
        self.CALPO = 50*np.ones((soil_layers,), dtype=int)	#Solid Chemistry - Variscite (g P Mg-1)
        self.CFEPO = 50*np.ones((soil_layers,), dtype=int)	#Solid Chemistry - Strengite (g P Mg-1)
        self.CCAPD = 50*np.ones((soil_layers,), dtype=int)	#Solid Chemistry -  Monetite (g P Mg-1)
        self.CCAPH = 50*np.ones((soil_layers,), dtype=int)	#Solid Chemistry - Hydroxyapatite (g P Mg-1)
        self.CALOH = 1000*np.ones((soil_layers,), dtype=int)	#Solid Chemistry - Amorphous aluminum hydroxide (g Al Mg-1)
        self.CFEOH = 2000*np.ones((soil_layers,), dtype=int)	#Solid Chemistry - Soil iron (g Fe Mg-1)
        self.CCACO = 0*np.ones((soil_layers,), dtype=int)	#Solid Chemistry - Calcite (g Ca Mg-1)
        self.CCASO = 0*np.ones((soil_layers,), dtype=int)	#Solid Chemistry - Gypsum (g Ca Mg-1)
        self.GKC4 = 0.025*np.ones((soil_layers,), dtype=int)	#Exchange Chemistry - Gapon selectivity coefficient for Ca2+ - NH4+ exchange
        self.GKCH = 0.25*np.ones((soil_layers,), dtype=int)	#Exchange Chemistry - Gapon selectivity coefficient for Ca2+ - H+  exchange
        self.GKCA = 0.25*np.ones((soil_layers,), dtype=int)	#Exchange Chemistry - Gapon selectivity coefficient for Ca2+ - Al3+ exchange
        self.GKCM = 0.6*np.ones((soil_layers,), dtype=int)	#Exchange Chemistry - Gapon selectivity coefficient for Ca2+ - Mg2+ exchange
        self.GKCN = 0.16*np.ones((soil_layers,), dtype=int)	#Exchange Chemistry - Gapon selectivity coefficient for Ca2+ - Na+ exchange
        self.GKCK = 3*np.ones((soil_layers,), dtype=int)	#Exchange Chemistry - Gapon selectivity coefficient for Ca2+ - K+ exchange
        self.THW = 1*np.ones((soil_layers,), dtype=int)	#Hydrology - Initial water content (m3 m-3)
        self.THI = 0*np.ones((soil_layers,), dtype=int)	#Hydrology - Initial ice content  (m3 m-3 water equivalent)
        self.RSC_1 = 45*np.ones((soil_layers,), dtype=int)	#Biology - Initial fine plant residue C (g C m-2)
        self.RSN_1 = 1.5*np.ones((soil_layers,), dtype=int)	#Biology - Initial fine plant residue N (g N m-2)
        self.RSP_1 = 0.15*np.ones((soil_layers,), dtype=int)	#Biology - Initial fine plant residue P (g P m-2)
        self.RSC_0 = 0*np.ones((soil_layers,), dtype=int)	#Biology -  Initial coarse woody residue C (g C m-2)
        self.RSN_0 = 0*np.ones((soil_layers,), dtype=int)	#Biology - Initial coarse woody residue N (g N m-2)
        self.RSP_0 = 0*np.ones((soil_layers,), dtype=int)	#Biology - Initial coarse woody residue P (g P m-2)
        self.RSC_2 = 0*np.ones((soil_layers,), dtype=int)	#Biology -  Initial Animal manure C (g C m-2)
        self.RSN_2 = 0*np.ones((soil_layers,), dtype=int)	#Biology -  Initial Animal manure N (g N m-2)
        self.RSP_2 = 0*np.ones((soil_layers,), dtype=int)	#Biology - Initial Animal manure P (g P m-2)
        
    def set_soil_parameters_surface(self,param_names,param_values):
        try:
            for i in range(len(param_names)):
                setattr(self, param_names[i], param_values[i])     
        except Exception as e:
            print(e)           
            
    def set_soil_parameters_profile(self,param_names,soil_depths,param_values):
        try:
            for i in range(len(param_names)):
                tmp = -1*np.ones((len(self.CDPTH)))
                for layers_i in range(len(self.CDPTH)):
                    for soil_depths_i in range(soil_depths.shape[0]):
                        if self.CDPTH[layers_i] > soil_depths[soil_depths_i,0] and \
                           self.CDPTH[layers_i] <= soil_depths[soil_depths_i,1]:                            
                            tmp[layers_i] = param_values[i,soil_depths_i]
                        if self.CDPTH[layers_i] > np.max(soil_depths[:,1]):
                            tmp[layers_i] = param_values[i,soil_depths[:,1]==np.max(soil_depths[:,1])]
                setattr(self, param_names[i], tmp)
        except Exception as e:
            print(e)           
            
    def write_soil_file(self, soil_file_output):
        try:
            fout = open(soil_file_output, "w")
            # write Line # 1
            fout.write((19*'%.6s,' + '%.6s\n') % (self.PSIFC, self.PSIWP, self.ALBS, self.PH_0, self.RSC_1_0, self.RSN_1_0,\
                                              self.RSP_1_0, self.RSC_0_0, self.RSN_0_0, self.RSP_0_0, self.RSC_2_0,\
                                              self.RSN_2_0, self.RSP_2_0, self.IXTYP_1, self.IXTYP_2, self.NU,\
                                              self.NJ, self.NL1, self.NL2, self.ISOILR))
            # write Line # 2
            for si in self.S_Column_Parameters:
                #print(si)
                fout.write(((self.NJ-1)*'%.6s,' + '%.6s\n') % (tuple(getattr(self, si)[:self.NJ])))
            fout.close()
        except Exception as e:
            print(e)     
    
    def read_soil_file(self, soil_file_input):
        fin = open(soil_file_input, "r")
        # read Line # 1
        self.PSIFC, self.PSIWP, self.ALBS, self.PH_0, self.RSC_1_0, self.RSN_1_0,\
            self.RSP_1_0, self.RSC_0_0, self.RSN_0_0, self.RSP_0_0, self.RSC_2_0,\
            self.RSN_2_0, self.RSP_2_0, self.IXTYP_1, self.IXTYP_2, self.NU,\
            self.NJ, self.NL1, self.NL2, self.ISOILR = fin.readline().rstrip().split(',')
        self.NJ = int(self.NJ)
        for ii in range(len(self.S_Column_Parameters)):
            si = self.S_Column_Parameters[ii]
            setattr(self, si, np.array(fin.readline().rstrip().split(',')))
            
    def revise_soil_file(self, revised_parameters_name,revise_parameters_values):
        for i in range(len(revised_parameters_name)):
            setattr(self, revised_parameters_name[i], revise_parameters_values[i])    
            
def getFeatures(gdf):
        """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
        import json
        return [json.loads(gdf.to_json())['features'][0]['geometry']]

def showRaster(df,out_tif):
    clipped = rasterio.open(out_tif)
    fig, ax = plt.subplots(figsize=(8, 6))
    df.plot(color=None,facecolor='none',edgecolor='red',linewidth = 2,ax=ax)
    show(clipped, ax=ax)
    ax.axis('off');    

def clip(df, src, outName):
    coords = getFeatures(df)
    clipped_array, clipped_transform = mask(dataset=src, shapes=coords, crop=True)
    

    out_meta = src.meta.copy()
    out_meta.update({"driver": "GTiff",
                     "height": clipped_array.shape[1],
                     "width": clipped_array.shape[2],
                     "transform": clipped_transform})
   
    with rasterio.open(outName, "w", **out_meta) as dest:
        dest.write(clipped_array)

def mkdir(path):
     if not os.path.exists(path):
         os.makedirs(path)

class extractSoilPropertyForOnePoint():
    def __init__(self,dfcop,dfcho,dfchf):
        self.cop_last = []
        self.component_key_last = []
        self.cop0_last = []
        self.dfcop = dfcop
        self.dfcho = dfcho
        self.dfchf = dfchf
        
    def run(self,MUKEY): 
        if MUKEY == 0:
            raise ValueError
           
        # extract the most representative sample: cop0 and cho0
        # find the crresponding cokey of the representative conponent using MUKEY
        # comppct_r means the percent of representative %
        cop=self.dfcop.loc[self.dfcop['mukey']==str(MUKEY)]
    
        component_key=self.dfcop['cokey'].values[cop['comppct_r'].idxmax()] 
        cop0=cop.loc[cop['cokey']==component_key]
        if self.dfcho.loc[self.dfcho['cokey']==component_key].shape[0]==0:
            cop=self.cop_last
            component_key=self.component_key_last
            cop0=self.cop0_last
        cho0=self.dfcho.loc[self.dfcho['cokey']==component_key]
        cho_r=cho0.shape[0]#soil layers  
        
        ## extract parameters of first soil layer, commented by Q
        # cho: the first layer , hzdept_r means Top Depth - Representative Value
        # chkey means chorizon key, hzdepb_r means Bottom Depth - Representative Value
    
        cho=self.dfcho.loc[cho0['hzdept_r'].idxmin()].to_frame().T
        # except:
        #     print(i)
        #     raise ValueError
        chf=self.dfchf.loc[self.dfchf['chkey']==cho['chkey'].values[0]]#the corresponding chfrags of chorizon
        soil_depths=np.array([pd.to_numeric(cho['hzdept_r']).iloc[0],pd.to_numeric(cho['hzdepb_r']).iloc[0]])   
        p=cho
        
        ## assign values to each soil attributes  
        # surface ALBSO: Albedo Dry - Representative Value
        ALBS0=pd.to_numeric(cop0['albedodry_r']).iloc[0]  #Soil albedo when soil surface dries
        if ALBS0 is None:
            ALBS0=0.2
            
        # physics
        BKDS=[pd.to_numeric(p['dbovendry_r']).iloc[0] for k in range(cho_r)]#Bulk density  (Mg m-3) Db oven dry - Representative Value
        CSAND=[pd.to_numeric(p['sandtotal_r']).iloc[0] for k in range(cho_r)]#sand content                
        CSILT=[pd.to_numeric(p['silttotal_r']).iloc[0] for k in range(cho_r)]#silt content 
        ROCK=[pd.to_numeric(sum(chf['fragvol_r'])) for k in range(cho_r)]#volume fraction occupied by coarse fragments   
        #hydrology
        FC=[pd.to_numeric(p['wthirdbar_r']).iloc[0] for k in range(cho_r)]# at field capacity
        WP=[pd.to_numeric(p['wfifteenbar_r']).iloc[0] for k in range(cho_r)]#water content at wilting point
        SCNV=[pd.to_numeric(p['ksat_r']).iloc[0] for k in range(cho_r)]#saturated hydraulic conductivity in the vertical directi
        #biology
        CORGC=[pd.to_numeric(p['om_r']).iloc[0] for k in range(cho_r)]#Total Organic C
        CORGP=[pd.to_numeric(p['pbray1_r']).iloc[0] for k in range(cho_r)]#Biology Organic P
        pd.isnull(CORGP)==-1
        #solution chemistry
        PH=[pd.to_numeric(p['ph1to1h2o_r']).iloc[0] for k in range(cho_r)]#ph
        #CNH4=[pd.to_numeric(p['sumbases_r']).iloc[0] for k in range(cho_r)]#exchangeable NH4+
        CPO4=[pd.to_numeric(p['ph2osoluble_r']).iloc[0] for k in range(cho_r)]#exchangeable P
        CAL=[pd.to_numeric(p['extral_r']).iloc[0] for k in range(cho_r)]#Aluminum
        #CFE=[pd.to_numeric(p['freeiron_r']).iloc[0] for k in range(cho_r)]#Iron
        #CCA=[pd.to_numeric(p['gypsum_r']).iloc[0] for k in range(cho_r)]#Calcium
        #CSO4=[pd.to_numeric(p['gypsum_r']).iloc[0] for k in range(cho_r)]#Sulfate            
        #solid chemistry
        CEC=[pd.to_numeric(p['cec7_r']).iloc[0] for k in range(cho_r)]#cation exchange capacity
        #CFEOH=[pd.to_numeric(p['feoxalate_r']).iloc[0] for k in range(cho_r)]#Soil Iron
        CCASO=[pd.to_numeric(p['gypsum_r']).iloc[0] for k in range(cho_r)]#Gypsum
        #exchange chemistry
        #nothing           
        
        #the second to the deepst layers
        if cho_r>1: 
            for j in range(1,cho_r):
                p=cho0.loc[cho0['hzdept_r']==cho['hzdepb_r'].values[j-1]]# match the largest value of this layer to the least value of the deepest layer
                # cho=cho.append(p,ignore_index=True)
                cho= pd.concat([cho,p],ignore_index=True)
                chf=self.dfchf.loc[self.dfchf['chkey']==cho['chkey'].values[j]]
                #各个粒径下大颗粒的体积比之和即为chfrags=sum of results of each grain size
                soil_depths=np.vstack((soil_depths,np.array([pd.to_numeric(p['hzdept_r']).iloc[0],pd.to_numeric(p['hzdepb_r']).iloc[0]])))
            
                ## input each soil attributes
                #surface
                #ALBS its value has been given before #Soil albedo when soil surface dries
                #physics
                BKDS[j]=pd.to_numeric(p['dbovendry_r']).iloc[0]     #bulk density
                CSAND[j]=pd.to_numeric(p['sandtotal_r']).iloc[0] #sand content
                CSILT[j]=pd.to_numeric(p['silttotal_r']).iloc[0] #silt content
                ROCK[j]=pd.to_numeric(sum(chf['fragvol_r']))#volume fraction occupied by coarse fragments
                #hydrology
                FC[j]=pd.to_numeric(p['wthirdbar_r']).iloc[0] #water content at field capacity
                WP[j]=pd.to_numeric(p['wfifteenbar_r']).iloc[0] #water content at wilting point
                SCNV[j]=pd.to_numeric(p['ksat_r']).iloc[0] #saturated hydraulic conductivity in the vertical direction
                #biology
                CORGC[j]=pd.to_numeric(p['om_r']).iloc[0] #Total Organic C
                CORGP[j]=pd.to_numeric(p['pbray1_r']).iloc[0] #Biology Organic P
                #solution chemistry
                PH[j]=pd.to_numeric(p['ph1to1h2o_r']).iloc[0] #ph
                #CNH4[j]=pd.to_numeric(p['sumbases_r']).iloc[0] #exchangeable NH4+
                CPO4[j]=pd.to_numeric(p['ph2osoluble_r']).iloc[0] #exchangeable P
                CAL[j]=pd.to_numeric(p['extral_r']).iloc[0] #Aluminum
                #CFE[j]=pd.to_numeric(p['freeiron_r']).iloc[0] #Iron
                #CCA[j]=pd.to_numeric(p['gypsum_r']).iloc[0] #Calcium
                #CSO4[j]=pd.to_numeric(p['gypsum_r']).iloc[0] #Sulfate            
                #solid chemistry
                CEC[j]=pd.to_numeric(p['cec7_r']).iloc[0] #cation exchange capacity
                #CFEOH[j]=pd.to_numeric(p['feoxalate_r']).iloc[0] #Soil Iron
                CCASO[j]=pd.to_numeric(p['gypsum_r']).iloc[0] #Gypsum
                #exchange chemistry
                #nothing
                
        #give initial values to those Nan value of each attribute
        for l in range(0,cho_r):
            if np.isnan(BKDS[l]): #bulk density
                BKDS[l]=-1
            if np.isnan(CSAND[l]): #sand content
                CSAND[l]=-1
            if np.isnan(CSILT[l]): #silt content
                CSILT[l]=-1
            if np.isnan(ROCK[l]):#volume fraction occupied by coarse fragments
                ROCK[l]=0
                #hydrology
            if np.isnan(FC[l]): #water content at field capacity
                FC[l]=-1
            if np.isnan(WP[l]): #water content at wilting point
                WP[l]=-1
            if np.isnan(SCNV[l]): #saturated hydraulic conductivity in the vertical direction
                SCNV[l]=-1
            #biology
            if np.isnan(CORGC[l]): #Total Organic C
                CORGC[l]=-1
            if np.isnan(CORGP[l]): #Biology Organic P
                CORGP[l]=-1
            #solution chemistry
            if np.isnan(PH[l]): #ph
                PH[l]=5.33
            #if np.isnan(CNH4[l]):#exchangeable NH4+
                #CNH4[l]=3
            if np.isnan(CPO4[l]): #exchangeable P
                CPO4[l]=10
            if np.isnan(CAL[l]):#Aluminum
                CAL[l]=-1
            #if np.isnan(CFE[l]): #Iron
                #CFE[l]=-1
            if np.isnan(CEC[l]):#cation exchange capacity
                CEC[l]=-1
            #if np.isnan(CFEOH[l]): #Soil Iron
                #CFEOH[l]=2000
            if np.isnan(CCASO[l]): #Gypsum
                CCASO[l]=0
    
        soil_depths=soil_depths*0.01#the unit of soil depth is meter
        #BKDS
        CSAND=[i*10 for i in CSAND]
        CSILT=[i*10 for i in CSILT]
        ROCK=[i/100 for i in ROCK]
        FC=[i/100 for i in FC]
        WP=[i/100 for i in WP]
        SCNV=[i*3.6 for i in SCNV]
        CORGC=[i*0.58*10 for i in CORGC]
        #CORGP
        #PH
        #CNH4=[i*180 for i in CNH4]
        #CPO4
        CAL=[i*270 for i in CAL]
        ##CFE
        #CEC
        ##CFEOH
        CCASO=[i*10**4 for i in CCASO]
       
        # buffer of the previous properties, if the next point get NaN, this buffer will be used
        self.cop_last = cop
        self.component_key_last = component_key
        self.cop0_last = cop0
        
        soil_prop_dic = {}
        soil_prop_dic['soil_depths'] = soil_depths
        soil_prop_dic['BKDS'] = BKDS
        soil_prop_dic['CSAND'] = CSAND
        soil_prop_dic['CSILT'] = CSILT
        soil_prop_dic['ROCK'] = ROCK
        
        #hydrology
        soil_prop_dic['FC'] = FC
        soil_prop_dic['WP'] = WP
        soil_prop_dic['SCNV'] = SCNV
       
        #biology
        soil_prop_dic['CORGC'] = CORGC
        soil_prop_dic['CORGP'] = CORGP
       
        #solution chemistry
        soil_prop_dic['PH'] = PH
        soil_prop_dic['CPO4'] = CPO4
        soil_prop_dic['CAL'] = CAL
        soil_prop_dic['CEC'] = CEC
        soil_prop_dic['CCASO'] = CCASO
        
        return soil_prop_dic

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
    
    # shpfile & in raster
    # shp_df = gpd.read_file('E:/shp/county_corn_belt_zhou_cut_prj_5070.shp')    # zhou wang's shp has displacements, shit!
    shp_df = gpd.read_file('I:/shp/shp_corn_belt_TIGER_5070.shp')
    id_field = 'GEOID' #'FIPS'
    # outdir = 'H:/My Drive/CDL_cornBelt' # has displacements, shit!
    outdir = 'I:/gSSURGO_data'
   
    yearSpan = [t for t in range(2000,2007+1)]
    inras = 'I:/gSSURGO_data/October 2021 gSSURGO CONUS/MapunitRaster_30m.tif'  
    
    # clip    
    FIPSlist = ['17019']
    
    for FIPS in FIPSlist:  # FIPS = FIPSlist[0]
        df = shp_df.loc[shp_df[id_field]==FIPS]       
        outName = '%s/county_30m/FIPS_%s.tif'%(outdir,FIPS)
        if not os.path.exists(outName):
            mkdir('%s/county_30m'%(outdir))
            src  = rasterio.open(inras)
            clip(df, src, outName)
            
        # stat of the MUKEY
        geoimg = gdal.Open(outName)
        gt_forward = geoimg.GetGeoTransform()
        img = geoimg.ReadAsArray()
        im_proj = geoimg.GetProjection()
        im_geotrans = geoimg.GetGeoTransform()
        # plt.figure()
        # plt.imshow(img)
        
        MUKEY_list = list(set(img[img>0]))
        
        # extract soil properties
        # param_names=np.array(['BKDS','CSAND','CSILT','ROCK','FC','WP','SCNV','CORGC','CORGP','PH','CPO4','CAL','CEC','CCASO'])
        # Bulk density, Field capacity, Wilting point, saturated hydraulic conductivity in the vertical direction (Ks),
        # Sand content,Silt content,Total organic carbon (SOC)
        param_names=['BKDS','FC','WP','SCNV','CSAND','CSILT','CORGC']
        start = time.time()
        print('loading gpd...')
        abb = 'IL'
        gSSURGO_location_state = r"I:\gSSURGO_data\October 2021 gSSURGO by State\gSSURGO_{}\gSSURGO_{}.gdb"
        gSSURGO_location = gSSURGO_location_state.format(abb,abb)
        dfcop=gpd.read_file(gSSURGO_location,layer='component')
        dfcho=gpd.read_file(gSSURGO_location,layer='chorizon')
        dfchf=gpd.read_file(gSSURGO_location,layer='chfrags')
        
        elapsed_time = time.time() - start
        print("reading gdb file takes {:.1f} sec".format(elapsed_time))
        #use three keys to find soil properities of each subfileds
        #the relationship of MUPOLYGON,Conponent,Chorizon and Chfrags are: MOPOLYGON_mukey_Conponent_cokey_Chorizon_chkey_Chfrags
        #mukey is the 4th row of MUPOLYGON  
             
        #initialize
        exSoil = extractSoilPropertyForOnePoint(dfcop,dfcho,dfchf)
        
        soil_prop_all = {}
        for MUKEY in MUKEY_list:
            soil_prop_dic = exSoil.run(MUKEY)
            soil_prop_all[MUKEY] = soil_prop_dic
            
        # generate soil property map
        for p in param_names:
            print('Processing %s'%p)
            para_img = np.zeros((img.shape[0],img.shape[1]))
            for MUKEY in MUKEY_list:
                tmp = soil_prop_all[MUKEY][p][0]  # only take the first layer
                para_img[img==MUKEY] = tmp
                
            para_img = para_img.astype(np.float32)
            # plt.figure()
            # plt.imshow(para_img)
            # creat geotif
            write_geo_tiff(im_proj, im_geotrans, img=para_img,
                           path='%s/county_30m/FIPS_%s_%s.tif'%(outdir,FIPS,p), dataType = 'Float32',NoData=0)
        