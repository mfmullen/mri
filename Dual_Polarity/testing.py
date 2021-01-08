# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 14:26:58 2020

@author: Mike
"""
import numpy as np
import scipy.io as scio
from distortion_correction import distortion_correction
import matplotlib.pyplot as plt

if ('fplus' not in locals()) or ('fminus' not in locals()):
    recon_orig = scio.loadmat('20201209_3T_data_rsos_full.mat')['recon_orig']
    recon_orig = recon_orig/np.max(recon_orig)
    recon_orig = np.transpose(recon_orig,(2,0,1,3))
    fminus = recon_orig[:,:,:,1].astype(np.float32)
    fplus = recon_orig[:,:,:,0].astype(np.float32)
    nx, ny, nz = fplus.shape

#plt.imshow(np.abs(fplus[:,:,128]))
#plt.imshow(np.abs(fminus[:,:,128]))
dc = distortion_correction(image_size=fplus.shape)

Tx, cost = dc.estimate_b0(fplus,fminus)
#%%

plt.figure(2)
b0plot = plt.imshow(Tx[:,128,:])
plt.colorbar()