# -*- coding: utf-8 -*-
"""
Created on Sun May  8 21:04:47 2022

@author: yang8460
"""
import torch
import torch.nn as nn
import numpy as np
from  torch.distributions import multivariate_normal

device = "cuda" if torch.cuda.is_available() else "cpu"


class KG_ecosys(nn.Module):
    '''
     v2: spped up the network runs
    '''
    def __init__(self, input_dim, hidden_dim, output_dim=[1,3,3,2],noise_cv=0.01, mode='paraPheno_c2'):
        super().__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.cv = noise_cv
        self.mode = mode
        self.parasList = ['CropType','VCMX','CHL4','GROUPX','STMX','GFILL','SLA1','Bulk density','Field capacity'
                          ,'Wilting point','Ks','Sand content','Silt content','SOC','Fertilizer']
        
        # GRU layers
        self.gruCell_1 = nn.GRUCell(input_size=self.input_dim, hidden_size=hidden_dim)
        self.gruCell_2 = nn.GRUCell(input_size=self.input_dim+1, hidden_size=hidden_dim)
        self.gruCell_3 = nn.GRUCell(input_size=hidden_dim*2, hidden_size=hidden_dim)
        self.gruCell_4 = nn.GRUCell(input_size=hidden_dim*2, hidden_size=hidden_dim)
        
        # Fully connected layer
        self.outDim = output_dim
        self.fc_1 = nn.Linear(hidden_dim, self.outDim[0])
        self.fc_2 = nn.Linear(hidden_dim, self.outDim[1])
        self.fc_3 = nn.Linear(hidden_dim, self.outDim[2])
        self.fc_4 = nn.Linear(hidden_dim, self.outDim[3])
        
        if self.mode == 'para':
            self.fc_1_v1 = nn.Linear(self.outDim[0]+len(self.parasList),32) # + paras
            self.fc_2_v1 = nn.Linear(self.outDim[1]+len(self.parasList),32) # + paras
            self.fc_3_v1 = nn.Linear(self.outDim[2]+len(self.parasList),32) # + paras
            self.fc_4_v1 = nn.Linear(self.outDim[3]+len(self.parasList),32) # + paras
  
        elif self.mode == 'basic':
            self.fc_1_v1 = nn.Linear(self.outDim[0],32) # 
            self.fc_2_v1 = nn.Linear(self.outDim[1],32) # 
            self.fc_3_v1 = nn.Linear(self.outDim[2],32) #
            self.fc_4_v1 = nn.Linear(self.outDim[3],32) # 

        elif self.mode == 'crop':
            self.fc_1_v1 = nn.Linear(self.outDim[0]+1,32) # + croptype
            self.fc_2_v1 = nn.Linear(self.outDim[1]+1,32) # + croptype
            self.fc_3_v1 = nn.Linear(self.outDim[2]+1,32) # + croptype
            self.fc_4_v1 = nn.Linear(self.outDim[3]+1,32) # + croptype
            
        elif self.mode == 'paraPheno_c2':
            self.fc_1_v1 = nn.Linear(self.outDim[0]+len(self.parasList),32) # + paras
            self.fc_2_v1 = nn.Linear(self.outDim[1]+len(self.parasList)+1,32) # + paras
            self.fc_3_v1 = nn.Linear(self.outDim[2]+len(self.parasList),32) # + paras
            self.fc_4_v1 = nn.Linear(self.outDim[3]+len(self.parasList),32) # + paras
        
        elif self.mode == 'cropPheno_c2':
            self.fc_1_v1 = nn.Linear(self.outDim[0]+1,32) # + croptype
            self.fc_2_v1 = nn.Linear(self.outDim[1]+2,32) # + croptype & pheno
            self.fc_3_v1 = nn.Linear(self.outDim[2]+1,32) # + croptype
            self.fc_4_v1 = nn.Linear(self.outDim[3]+1,32) # + croptype
            
        self.fc_1_v2 = nn.Linear(32,32)
        self.fc_2_v2 = nn.Linear(32,32)
        self.fc_3_v2 = nn.Linear(32,32)
        self.fc_4_v2 = nn.Linear(32,32)
        
        self.fc_1_v3 = nn.Linear(32,hidden_dim)
        self.fc_2_v3 = nn.Linear(32,hidden_dim)
        self.fc_3_v3 = nn.Linear(32,hidden_dim)
        self.fc_4_v3 = nn.Linear(32,hidden_dim)
        
        self.relu = nn.ReLU()
        
        # Bn of inputs
        self.bn = nn.BatchNorm1d(input_dim)

    
    def forward(self, x, hidden=None, initLAI=None, isTrain=False, updateState=None, seq_lengthList=None):
        self.batchsize = x.size(0)
        
        # initialize inputs
        if hidden==None:
            # Initializing hidden state for first input with zeros
            self.hidden = []
            for i in range(4):
                h0 = torch.zeros(x.size(0), self.hidden_dim).requires_grad_().to(device)
                self.hidden.append(h0.detach())
        else:
            self.hidden = [h.detach() for h in hidden]
        if initLAI==None:
            self.LAI_previous = torch.zeros(x.size(0), 1).to(device).detach()
        else:
            self.LAI_previous = initLAI.detach()
            
        # deploy BN
        # x_debug = torch.clone(x.detach())
        # x_debug_n = x_debug.detach().cpu().numpy()
        x = x.permute(0, 2, 1)        
        x = self.bn(x)
        x = x.permute(0, 2, 1)
        # x_debug_n_bn = x.detach().cpu().numpy()
        self.hidden_1,self.hidden_2,self.hidden_3,self.hidden_4 = self.hidden

        # Forward propagation by passing in the input and hidden state into the model
        # loop for sequence length
        # RNNcell cannot receive packed sequence, so we didn't pack them, it will take more comperting but it is ok
        out1 = torch.zeros((self.batchsize,x.shape[1],self.outDim[0])).to(device)
        out2 = torch.zeros((self.batchsize,x.shape[1],self.outDim[1])).to(device)
        out3 = torch.zeros((self.batchsize,x.shape[1],self.outDim[2])).to(device)
        out4 = torch.zeros((self.batchsize,x.shape[1],self.outDim[3])).to(device)
        hTensor1 = torch.zeros((self.batchsize,x.shape[1],self.hidden_dim)).to(device)
        hTensor2 = torch.zeros((self.batchsize,x.shape[1],self.hidden_dim)).to(device)
        hTensor3 = torch.zeros((self.batchsize,x.shape[1],self.hidden_dim)).to(device)
        hTensor4 = torch.zeros((self.batchsize,x.shape[1],self.hidden_dim)).to(device)
        hTensor1_v = torch.zeros((self.batchsize,x.shape[1],self.hidden_dim)).to(device)
        hTensor2_v = torch.zeros((self.batchsize,x.shape[1],self.hidden_dim)).to(device)
        hTensor3_v = torch.zeros((self.batchsize,x.shape[1],self.hidden_dim)).to(device)
        hTensor4_v = torch.zeros((self.batchsize,x.shape[1],self.hidden_dim)).to(device)

        for i in range(x.shape[1]):
            
            # update hidden state
            if not (updateState is None):
                self.stateC1 = updateState[:,0:self.outDim[0]]
                self.stateC2 = updateState[:,self.outDim[0]:np.sum(self.outDim[0:2])]
                self.stateC3 = updateState[:,np.sum(self.outDim[0:2]):np.sum(self.outDim[0:3])]
                self.stateC4 = updateState[:,np.sum(self.outDim[0:3]):np.sum(self.outDim[0:4])]

                cropType = x[:,i,:][:,6].view([-1,1])
                parameter = x[:,i,:][:,6:]
                
                # cropType_d = x_debug[:,i,:][:,6].view([-1,1]).detach().cpu().numpy()
                # parameter_d = x_debug[:,i,:][:,6:-1].detach().cpu().numpy()
                # GDD_d =x_debug[:,i,:][:,-1].view([-1,1]).detach().cpu().numpy()
                
                if not np.isnan(self.stateC1).any():
                    self.stateC1 = torch.tensor(self.stateC1.astype(np.float32)).view([self.batchsize,-1]).to(device)
                    if self.mode == 'basic':
                        updateIn = self.stateC1.detach()
                    elif self.mode in ['para','paraPheno_c2']:    
                        updateIn = torch.cat([self.stateC1.detach(),parameter.detach()], dim=1)      

                    elif self.mode in ['crop','cropPheno_c2']:
                        updateIn = torch.cat([self.stateC1.detach(),cropType.detach()], dim=1)
                    
                        
                    self.hidden_1_v = self.fc_1_v3(self.relu(self.fc_1_v2(self.relu(self.fc_1_v1(updateIn)))))
                    self.hidden_1 = self.hidden_1_v
                else:
                    self.stateC1 = self.fc_1(self.hidden_1)
                    
                if not np.isnan(self.stateC2).any():
                    self.stateC2 = torch.tensor(self.stateC2.astype(np.float32)).view([self.batchsize,-1]).to(device)
                    if self.mode == 'basic':
                        updateIn = self.stateC2.detach()                                        
                    elif self.mode == 'para':
                        updateIn = torch.cat([self.stateC2.detach(),parameter.detach()], dim=1)
                    elif self.mode == 'crop':
                        updateIn = torch.cat([self.stateC2.detach(),cropType.detach()], dim=1)
                    elif self.mode == 'cropPheno_c2':
                        updateIn = torch.cat([self.stateC2.detach(),cropType.detach(),self.stateC1.detach()], dim=1)
                    elif self.mode == 'paraPheno_c2':
                        updateIn = torch.cat([self.stateC2.detach(),parameter.detach(),self.stateC1.detach()], dim=1)                   
                        
                    self.hidden_2_v = self.fc_2_v3(self.relu(self.fc_2_v2(self.relu(self.fc_2_v1(updateIn)))))
                    self.hidden_2 = self.hidden_2_v
                    
                if not np.isnan(self.stateC3).any():
                    self.stateC3 = torch.tensor(self.stateC3.astype(np.float32)).view([self.batchsize,-1]).to(device)
                    if self.mode == 'basic':
                        updateIn = self.stateC3.detach()
                  
                    elif self.mode in ['para','paraPheno_c2']:   
                        updateIn = torch.cat([self.stateC3.detach(),parameter.detach()], dim=1)
                    elif self.mode in ['crop','cropPheno_c2']:
                        updateIn = torch.cat([self.stateC3.detach(),cropType.detach()], dim=1)
                        
                    self.hidden_3_v = self.fc_3_v3(self.relu(self.fc_3_v2(self.relu(self.fc_3_v1(updateIn)))))
                    self.hidden_3 = self.hidden_3_v
                    
                if not np.isnan(self.stateC4).any():
                    self.stateC4 = torch.tensor(self.stateC4.astype(np.float32)).view([self.batchsize,-1]).to(device)
                    if self.mode == 'basic':
                        updateIn = self.stateC4.detach()
                  
                    elif self.mode in ['para','paraPheno_c2']:   
                        updateIn = torch.cat([self.stateC4.detach(),parameter.detach()], dim=1)
                    elif self.mode in ['crop','cropPheno_c2']:
                        updateIn = torch.cat([self.stateC4.detach(),cropType.detach()], dim=1)                   
                        
                    self.hidden_4_v = self.fc_4_v3(self.relu(self.fc_4_v2(self.relu(self.fc_4_v1(updateIn)))))
                    self.hidden_4 = self.hidden_4_v
            
            x_forward = x[:,i,:]
            self.hidden_1 = self.gruCell_1(x_forward, self.hidden_1)
            self.hidden_2 = self.gruCell_2(torch.cat([x_forward,self.LAI_previous],dim=1), self.hidden_2)
            cell1_out = self.fc_1(self.hidden_1)
            cell2_out = self.fc_2(self.hidden_2)
            
            self.hidden_3 = self.gruCell_3(torch.cat([self.hidden_1.detach(),self.hidden_2.detach()],dim=1), self.hidden_3)
            cell3_out = self.fc_3(self.hidden_3)
            
            self.hidden_4 = self.gruCell_4(torch.cat([self.hidden_1.detach(),self.hidden_3.detach()],dim=1), self.hidden_4)
            cell4_out = self.fc_4(self.hidden_4)
            self.LAI_previous = cell4_out[:,0].view([-1,1]).detach()
            
            out1[:,i,:] = cell1_out
            out2[:,i,:] = cell2_out
            out3[:,i,:] = cell3_out
            out4[:,i,:] = cell4_out
            
            if isTrain:
                ## cal noise
                # cell1
                cell1_u = torch.mean(cell1_out.detach(),dim=0).cpu().item()
                noise1 = torch.normal(mean=0,std=np.abs(cell1_u)*self.cv,size=(self.batchsize,1)).to(device)
                
                # cell2
                cell2_u = torch.mean(cell2_out.detach(),dim=0).cpu().numpy()
                m = np.zeros(self.outDim[1])
                tao = np.abs(cell2_u)*self.cv
                P = np.diag(tao**2)
                noise2 = torch.tensor(np.random.multivariate_normal(mean=m, cov=P, size=self.batchsize).astype(np.float32)).to(device)
                
                # cell3
                cell3_u = torch.mean(cell3_out.detach(),dim=0).cpu().numpy()
                m = np.zeros(self.outDim[2])
                tao = np.abs(cell3_u)*self.cv
                P = np.diag(tao**2)
                noise3 = torch.tensor(np.random.multivariate_normal(mean=m, cov=P, size=self.batchsize).astype(np.float32)).to(device)
                
                # cell4
                cell4_u = torch.mean(cell4_out.detach(),dim=0).cpu().numpy()
                m = np.zeros(self.outDim[3])
                tao = np.abs(cell4_u)*self.cv
                P = np.diag(tao**2)
                noise4 = torch.tensor(np.random.multivariate_normal(mean=m, cov=P, size=self.batchsize).astype(np.float32)).to(device)
                
                # # para
                parameter = x[:,i,:][:,6:]
                para_u = torch.mean(parameter.detach(),dim=0).cpu().numpy()
                m = np.zeros(parameter.shape[1])
                tao = np.abs(para_u)*self.cv
                P = np.diag(tao**2)
                noise_para = torch.tensor(np.random.multivariate_normal(mean=m, cov=P, size=self.batchsize).astype(np.float32)).to(device)
                
                # decoder input
                cropType = x[:,i,:][:,6].view([-1,1])
                                
                # phenology
                pheno = cell1_out.detach()
                
                # add Gaussian noise

                if self.mode == 'para':

                    self.hidden_1_v = self.fc_1_v3(self.relu(self.fc_1_v2(self.relu(self.fc_1_v1(torch.cat([cell1_out.detach() + noise1.detach(), 
                                                                                     parameter.detach() + noise_para.detach()],dim=1))))))
                    self.hidden_2_v = self.fc_2_v3(self.relu(self.fc_2_v2(self.relu(self.fc_2_v1(torch.cat([cell2_out.detach() + noise2.detach(), 
                                                                                     parameter.detach() + noise_para.detach()],dim=1))))))
                    self.hidden_3_v = self.fc_3_v3(self.relu(self.fc_3_v2(self.relu(self.fc_3_v1(torch.cat([cell3_out.detach() + noise3.detach(), 
                                                                                     parameter.detach() + noise_para.detach()],dim=1))))))
                    self.hidden_4_v = self.fc_4_v3(self.relu(self.fc_4_v2(self.relu(self.fc_4_v1(torch.cat([cell4_out.detach() + noise4.detach(), 
                                                                                     parameter.detach() + noise_para.detach()],dim=1))))))
                elif self.mode == 'paraPheno_c2':

                    self.hidden_1_v = self.fc_1_v3(self.relu(self.fc_1_v2(self.relu(self.fc_1_v1(torch.cat([cell1_out.detach() + noise1.detach(), 
                                                                                     parameter.detach() + noise_para.detach()],dim=1))))))
                    self.hidden_2_v = self.fc_2_v3(self.relu(self.fc_2_v2(self.relu(self.fc_2_v1(torch.cat([cell2_out.detach() + noise2.detach(), 
                                                                                     parameter.detach() + noise_para.detach(),pheno+ noise1.detach()],dim=1))))))
                    self.hidden_3_v = self.fc_3_v3(self.relu(self.fc_3_v2(self.relu(self.fc_3_v1(torch.cat([cell3_out.detach() + noise3.detach(), 
                                                                                     parameter.detach() + noise_para.detach()],dim=1))))))
                    self.hidden_4_v = self.fc_4_v3(self.relu(self.fc_4_v2(self.relu(self.fc_4_v1(torch.cat([cell4_out.detach() + noise4.detach(), 
                                                                                     parameter.detach() + noise_para.detach()],dim=1))))))
                    
                elif self.mode == 'basic':
 
                    self.hidden_1_v = self.fc_1_v3(self.relu(self.fc_1_v2(self.relu(self.fc_1_v1(torch.cat([cell1_out.detach() + noise1.detach()],dim=1))))))
                    self.hidden_2_v = self.fc_2_v3(self.relu(self.fc_2_v2(self.relu(self.fc_2_v1(torch.cat([cell2_out.detach() + noise2.detach()],dim=1))))))
                    self.hidden_3_v = self.fc_3_v3(self.relu(self.fc_3_v2(self.relu(self.fc_3_v1(torch.cat([cell3_out.detach() + noise3.detach()],dim=1))))))
                    self.hidden_4_v = self.fc_4_v3(self.relu(self.fc_4_v2(self.relu(self.fc_4_v1(torch.cat([cell4_out.detach() + noise4.detach()],dim=1))))))
                    
                elif self.mode == 'crop':

                    self.hidden_1_v = self.fc_1_v3(self.relu(self.fc_1_v2(self.relu(self.fc_1_v1(torch.cat([cell1_out.detach() + noise1.detach(), 
                                                                                     cropType.detach()],dim=1))))))
                    self.hidden_2_v = self.fc_2_v3(self.relu(self.fc_2_v2(self.relu(self.fc_2_v1(torch.cat([cell2_out.detach() + noise2.detach(), 
                                                                                     cropType.detach()],dim=1))))))
                    self.hidden_3_v = self.fc_3_v3(self.relu(self.fc_3_v2(self.relu(self.fc_3_v1(torch.cat([cell3_out.detach() + noise3.detach(), 
                                                                                     cropType.detach()],dim=1))))))
                    self.hidden_4_v = self.fc_4_v3(self.relu(self.fc_4_v2(self.relu(self.fc_4_v1(torch.cat([cell4_out.detach() + noise4.detach(), 
                                                                                     cropType.detach()],dim=1))))))
                elif self.mode == 'cropPheno_c2':

                    self.hidden_1_v = self.fc_1_v3(self.relu(self.fc_1_v2(self.relu(self.fc_1_v1(torch.cat([cell1_out.detach() + noise1.detach(), 
                                                                                     cropType.detach()],dim=1))))))
                    self.hidden_2_v = self.fc_2_v3(self.relu(self.fc_2_v2(self.relu(self.fc_2_v1(torch.cat([cell2_out.detach() + noise2.detach(), 
                                                                                     cropType.detach(),pheno+ noise1.detach()],dim=1))))))
                    self.hidden_3_v = self.fc_3_v3(self.relu(self.fc_3_v2(self.relu(self.fc_3_v1(torch.cat([cell3_out.detach() + noise3.detach(), 
                                                                                     cropType.detach()],dim=1))))))
                    self.hidden_4_v = self.fc_4_v3(self.relu(self.fc_4_v2(self.relu(self.fc_4_v1(torch.cat([cell4_out.detach() + noise4.detach(), 
                                                                                     cropType.detach()],dim=1))))))
                    
                hTensor1[:,i,:] = self.hidden_1
                hTensor2[:,i,:] = self.hidden_2
                hTensor3[:,i,:] = self.hidden_3
                hTensor4[:,i,:] = self.hidden_4
                
                hTensor1_v[:,i,:] = self.hidden_1_v
                hTensor2_v[:,i,:] = self.hidden_2_v
                hTensor3_v[:,i,:] = self.hidden_3_v
                hTensor4_v[:,i,:] = self.hidden_4_v          
        
        out = torch.cat([out1,out2,out3,out4],dim=2)
        
        if isTrain:    
            return out, [self.hidden_1,self.hidden_2,self.hidden_3,self.hidden_4],\
                [hTensor1,hTensor2,hTensor3,hTensor4],[hTensor1_v,hTensor2_v,hTensor3_v,hTensor4_v]
        else:
            return out, [self.hidden_1,self.hidden_2,self.hidden_3,self.hidden_4]

