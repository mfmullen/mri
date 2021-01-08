# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 10:06:00 2020

@author: Mike
"""
import numpy as np
import scipy.io as scio
from scipy.fft import fft, ifft, fftshift, ifftshift
import matplotlib.pyplot as plt
from grappa_2d import grappa_2d

if 'kspace' not in locals():
    kspace = scio.loadmat('20201209_3T_data_reduced.mat')['rawfid']
    ny, nx, nz, nd, nc = kspace.shape
    kspace = np.reshape(kspace,(ny, nx, nz, nd*nc))
    kspace = np.transpose(kspace,(3,0,1,2))
    kspace = ifftshift(ifft(ifftshift(kspace,axes=3),axis=3,norm='ortho'),axes=3)
    kspace = kspace[:,:,:,63]
if 'corr_mtx' not in locals():
    corr_mtx = scio.loadmat('20201209_3T_data_reduced.mat')['corr_mtx']


acs_halfwidth = 12
af = 3
af2 = 2

#k_acs = np.zeros((nc,2*acs_halfwidth,2*acs_halfwidth))
yrange = slice(int(np.floor(ny/2)-acs_halfwidth),int(np.floor(ny/2)+acs_halfwidth))
xrange = slice(int(np.floor(nx/2)-acs_halfwidth),int(np.floor(nx/2)+acs_halfwidth))

im_acs = kspace[:,yrange,xrange]
for idx in range(1,3):
    im_acs = ifftshift(im_acs,axes=idx)
    im_acs = ifft(im_acs,axis=idx,norm='ortho')
    im_acs = ifftshift(im_acs,axes=idx)
    
im_full = kspace
for idx in range(1,3):
    im_full = ifftshift(im_full,axes=idx)
    im_full = ifft(im_full,axis=idx,norm='ortho')
    im_full = ifftshift(im_full,axes=idx)
    
im_reduced = kspace[:,::af,::af2]

for idx in range(1,3):
    im_reduced = ifftshift(im_reduced,axes=idx)
    im_reduced = ifft(im_reduced,axis=idx,norm='ortho')
    im_reduced = ifftshift(im_reduced,axes=idx)

grappa_kspace = grappa_2d(k_acs = im_acs, im_reduced = im_reduced, af = af, af2 = af2, corr_mtx = corr_mtx)
recon, g = grappa_kspace.run_recon()
plt.figure(1)
reconplot = plt.imshow(np.abs(recon[1,:,:]))
reconplot.set_cmap('gray')

plt.figure(2)
plt.imshow(g)
plt.colorbar()
