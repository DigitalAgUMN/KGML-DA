# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 12:25:12 2022

@author: yang8460
v2: Torch version of EnKF, 44s to 25s， ps: cov() and inv() still need loop
v3: Torch version of EnKF, 25s to 14s， ps: inv() still need loop
v4: no loop
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import torch
from copy import deepcopy
import numpy as np
from numpy import  zeros, eye
device = "cuda" if torch.cuda.is_available() else "cpu"
        
class EnsembleKalmanFilter_parallel_v4(object):

    def __init__(self, x, P, dim_z, N, H, fx, cellRange=None):
        
        if dim_z <= 0:
            raise ValueError('dim_z must be greater than zero')

        if N <= 0:
            raise ValueError('N must be greater than zero')

        dim_x = len(x)
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.N = N
        self.H = H.T

        self.fx = fx

        # self.Q = eye(dim_x)       # process uncertainty # discarded by Qi
        self.R = eye(dim_z)       # state uncertainty
        self.inv = np.linalg.inv

        # used to create error terms centered at 0 mean for
        # state and measurement
        self._mean = zeros(dim_x)
        self._mean_z = zeros(dim_z)
        self.cellRange = cellRange
        
    def torch_dot3d(self,A,B):
        return torch.einsum('ijk,ikl->ijl', A, B)

    def torch_dot3d_2d(self,A,B):
        return torch.einsum('ijk,kl->ijl', A, B)
    
    def torch_dot2d_3d(self,A,B):
        return torch.einsum('lj,ijk->ilk', A, B)
    
    def torch_cov_3d(self,input_mat):
        mean_mat = torch.mean(input_mat,axis=1,keepdims=True)
        x_mat_diff = input_mat- mean_mat
        cov_matrix = self.torch_dot3d(x_mat_diff.transpose(2,1), x_mat_diff) / (input_mat.shape[1]-1)
        return cov_matrix

    @torch.no_grad()
    def update(self, zList, R=None, Q = None, MaskedIndex = None, MaskedSample = None):  # 'Q': the process error, added by Qi 2020/10/9
        """
        Add a new measurement (z) to the kalman filter. If z is None, nothing
        is changed.

        Parameters
        ----------

        z : np.array
            measurement for this update.

        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise self.R will be used.
        MaskedIndex : don't update this state variable
        
        """

        if zList is None:
            return
        zList = torch.tensor(np.array(zList), dtype=torch.float32).to(device)
        if R is None:
            R = self.R
        if np.isscalar(R):
            R = eye(self.dim_z) * R
        if Q is None:  # 'Q': the process error, added by Qi 2020/10/9
            Q = R.copy()
            
        R = torch.tensor(R, dtype=torch.float32).to(device)
        N = self.N
  
        self.K_batch = []
        self.x_post_batch = []
        self.P_post_batch = []
        
        # calculate Kalman Gain
        PHT_3d = self.torch_dot3d_2d(torch.clone(self.P_batch),self.H.T)
        S_3d = self.torch_dot2d_3d(self.H, PHT_3d) + R # (dim_z, dim_z) or (batch,dim_z,dim_z)
        SI_3d = torch.linalg.inv(S_3d)
        K_3d = self.torch_dot3d(PHT_3d, SI_3d)
        
        # mask out unupdated state
        if not (MaskedIndex is None):
            for m in MaskedIndex:
                K_3d[:,m,:] = 0
                
        # mask out samples with nan obs
        if not (MaskedSample is None):
            K_3d[MaskedSample,:,:] = 0
            
        # update sigmas
        if len(Q.shape)>2:
            Q = np.mean(Q,axis=0)
        e = torch.tensor(np.random.multivariate_normal(mean=self._mean_z, cov=Q, size=N).astype(np.float32).transpose(1,0)[np.newaxis,:,:]).to(device)
        diff_sigmas_3d = torch.unsqueeze(zList,dim=2) - self.torch_dot2d_3d(self.H, torch.clone(self.sigmas_batch).transpose(2,1)) + e
        self.sigmas_batch +=  self.torch_dot3d(K_3d, diff_sigmas_3d).transpose(2,1)
        self.x_batch = torch.mean(self.sigmas_batch, dim=1)
        self.P_batch -= self.torch_dot3d(self.torch_dot3d(K_3d, S_3d), K_3d.transpose(2,1))
               
        self.K_batch = torch.clone(K_3d.detach())
        self.x_post_batch = torch.clone(self.x_batch.detach())
        self.P_post_batch = torch.clone(self.P_batch.detach())

    @torch.no_grad()
    def predict(self,dailyIn,updateHidden = False, MaskCells = None, zP=False):
        """ Predict next position. """
        # run model
        if updateHidden:
            upstate = self.sigmas_batch.detach().cpu().numpy()
            if not (MaskCells is None):
                for m in MaskCells:
                    upstate[:,:,self.cellRange[m][0]:self.cellRange[m][1]] = np.nan
                    
            self.sigmasList = self.fx(dailyIn,upstate)
        else:
            self.sigmasList = self.fx(dailyIn)
                            
        # calculate xf and Pf
        if len(self.sigmasList) == 1:
            self.sigmas_batch = torch.unsqueeze(torch.squeeze(torch.stack(self.sigmasList)),dim=0)
        else:
            self.sigmas_batch = torch.squeeze(torch.stack(self.sigmasList))
            
        self.x_batch = torch.mean(self.sigmas_batch,dim=1)        
        self.P_batch = self.torch_cov_3d(self.sigmas_batch)
        
        # calculate the P of measurement for std adaption
        self.sigmas_z_batch = self.torch_dot3d_2d(self.sigmas_batch,self.H.T)
        self.Pz_batch = self.torch_cov_3d(self.sigmas_z_batch)
            
        # save prior
        self.x_prior_batch = torch.clone(self.x_batch.detach())
        self.P_prior_batch = torch.clone(self.P_batch.detach())
        
class EnsembleKalmanFilter_parallel_v4_UpdatePara(object):

    def __init__(self, x, P, dim_z, N, H, fx, cellRange=None):
        
        if dim_z <= 0:
            raise ValueError('dim_z must be greater than zero')

        if N <= 0:
            raise ValueError('N must be greater than zero')

        dim_x = len(x)
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.N = N
        self.H = H.T

        self.fx = fx

        # self.Q = eye(dim_x)       # process uncertainty # discarded by Qi
        self.R = eye(dim_z)       # state uncertainty
        self.inv = np.linalg.inv

        # used to create error terms centered at 0 mean for
        # state and measurement
        self._mean = zeros(dim_x)
        self._mean_z = zeros(dim_z)
        self.cellRange = cellRange
        
    def torch_dot3d(self,A,B):
        return torch.einsum('ijk,ikl->ijl', A, B)

    def torch_dot3d_2d(self,A,B):
        return torch.einsum('ijk,kl->ijl', A, B)
    
    def torch_dot2d_3d(self,A,B):
        return torch.einsum('lj,ijk->ilk', A, B)
    
    def torch_cov_3d(self,input_mat):
        mean_mat = torch.mean(input_mat,axis=1,keepdims=True)
        x_mat_diff = input_mat- mean_mat
        cov_matrix = self.torch_dot3d(x_mat_diff.transpose(2,1), x_mat_diff) / (input_mat.shape[1]-1)
        return cov_matrix

    @torch.no_grad()
    def update(self, zList, R=None, Q = None, MaskedIndex = None):  # 'Q': the process error, added by Qi 2020/10/9
        """
        Add a new measurement (z) to the kalman filter. If z is None, nothing
        is changed.

        Parameters
        ----------

        z : np.array
            measurement for this update.

        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise self.R will be used.
        MaskedIndex : don't update this state variable
        
        """

        if zList is None:
            return
        zList = torch.tensor(np.array(zList), dtype=torch.float32).to(device)
        if R is None:
            R = self.R
        if np.isscalar(R):
            R = eye(self.dim_z) * R
        if Q is None:  # 'Q': the process error, added by Qi 2020/10/9
            Q = R.copy()
            
        R = torch.tensor(R, dtype=torch.float32).to(device)
        N = self.N
  
        self.K_batch = []
        self.x_post_batch = []
        self.P_post_batch = []
        
        # calculate Kalman Gain
        PHT_3d = self.torch_dot3d_2d(torch.clone(self.P_batch),self.H.T)
        S_3d = self.torch_dot2d_3d(self.H, PHT_3d) + R # (dim_z, dim_z)
        SI_3d = torch.linalg.inv(S_3d)
        K_3d = self.torch_dot3d(PHT_3d, SI_3d)
        
        # mask out unupdated state
        if not (MaskedIndex is None):
            for m in MaskedIndex:
                K_3d[:,m,:] = 0
                
        # update sigmas
        diff_sigmas_3d = torch.unsqueeze(zList,dim=2) - self.torch_dot2d_3d(self.H, torch.clone(self.sigmas_batch).transpose(2,1))
        self.sigmas_batch +=  self.torch_dot3d(K_3d, diff_sigmas_3d).transpose(2,1)
        self.x_batch = torch.mean(self.sigmas_batch, dim=1)
        self.P_batch -= self.torch_dot3d(self.torch_dot3d(K_3d, S_3d), K_3d.transpose(2,1))
               
        self.K_batch = torch.clone(K_3d.detach())
        self.x_post_batch = torch.clone(self.x_batch.detach())
        self.P_post_batch = torch.clone(self.P_batch.detach())

    def setPara(self,dailyIn, paraIndex = [7,8,9,10,11,12]):
        self.para = torch.from_numpy(dailyIn[:,:,0,paraIndex]).to(device)

    @torch.no_grad()
    def predict(self,dailyIn,updateHidden = False, MaskCells = None, stateIndex = [2], paraIndex = [7,8,9,10,11,12]):
        """ Predict next position. """
        # run model
        if updateHidden:
            dailyInUpdate = np.squeeze(dailyIn.copy())
            self.para = self.sigmas_batch[:,:,-len(paraIndex):]
            dailyInUpdate[:,:,paraIndex] = self.para.detach().cpu().numpy()
            dailyInUpdate = dailyInUpdate[:,:,np.newaxis,:]
            
            upstate = torch.squeeze(torch.stack(self.out)).detach().cpu().numpy()
            upstate[:,:,stateIndex] = self.sigmas_batch[:,:,:len(stateIndex)].detach().cpu().numpy()
            if not (MaskCells is None):
                for m in MaskCells:
                    upstate[:,:,self.cellRange[m][0]:self.cellRange[m][1]] = np.nan
                    
            self.out = self.fx(dailyInUpdate,upstate)
            
        else:
            dailyInUpdate = np.squeeze(dailyIn.copy())
            dailyInUpdate[:,:,paraIndex] = self.para.detach().cpu().numpy()
            dailyInUpdate = dailyInUpdate[:,:,np.newaxis,:]
            self.out = self.fx(dailyInUpdate)
            
        state = torch.squeeze(torch.stack(self.out)[:,:,:,stateIndex])
        if len(state.shape)==2:
            state = torch.unsqueeze(state,dim=2)
        self.sigmas_batch = torch.concat([state,self.para],axis=2)
            
        self.x_batch = torch.mean(self.sigmas_batch,dim=1)        
        self.P_batch = self.torch_cov_3d(self.sigmas_batch)
        
        # save prior
        self.x_prior_batch = torch.clone(self.x_batch.detach())
        self.P_prior_batch = torch.clone(self.P_batch.detach())