# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 14:26:58 2020

@author: Mike
"""
import numpy as np
#import cupy as cp
import scipy.io as scio
from distortion_correction_gpu import distortion_correction
import matplotlib.pyplot as plt
#cp.cuda.Device(1).use()
import time

if ('fplus' not in locals()) or ('fminus' not in locals()):
    recon_orig = scio.loadmat('20201209_3T_data_rsos_full.mat')['recon_orig']
    recon_orig = recon_orig/np.max(recon_orig)
    recon_orig = np.transpose(recon_orig,(2,0,1,3))
    fminus = np.asarray(recon_orig[:,:,:,1].astype(np.float32))
    fplus = np.asarray(recon_orig[:,:,:,0].astype(np.float32))
    nx, ny, nz = fplus.shape

#plt.imshow(np.abs(fplus[:,:,128]))
#plt.imshow(np.abs(fminus[:,:,128]))
dc = distortion_correction(image_size=fplus.shape)
#%%
start_time = time.time()
Ic, Tx = dc.run(fplus,fminus)
compute_time = time.time() - start_time
print(f'Data transfer + B0 estimation took {compute_time:.2f} seconds')
np.savez('20201209_3T_data_rsos_full_processed',Ic=Ic,Tx=Tx)

#164.563 seconds for all float16
#float32 took 163.005 seconds
#%%

#plt.figure(2)
#b0plot = plt.imshow(Tx[:,128,:])
#plt.colorbar()