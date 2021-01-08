# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 16:16:38 2020

@author: Mike
"""

import numpy as np
from scipy.fft import fftshift, ifftshift, fft, ifft
from scipy.linalg import inv, svd, lstsq
from scipy.linalg import norm as norm2
from scipy.signal import tukey

class grappa_2d:
    '''
    Documentation here
    '''
    def __init__(self, k_acs, im_reduced, af = 2, af2 = 1, caipi=0, corr_mtx = -1,srcx=5,srcy=4):
        nc, nx, ny = im_reduced.shape
        nx_acs, ny_acs = k_acs.shape[1:3]
        
        if corr_mtx.any() == -1:
            print('No correlation matrix passed: assuming perfect, uncorrelated coils.')
            corr_mtx = np.identity(nc)
    
        self.af = af
        self.af2 = af2
        self.caipi = caipi
        self.corr_mtx = corr_mtx
        self.srcx = srcx
        self.srcy = srcy
        self.k_acs = k_acs
        self.im_reduced = im_reduced
    
    def run_recon(self): 
    
        imsrc = self.k_acs
        imtrg = self.k_acs
        af = self.af
        af2 = self.af2
        corr_mtx = self.corr_mtx
        imfold = self.im_reduced
        
        nc, nyr,nxr = imfold.shape
        ncsrc, nysrc, nxsrc = imsrc.shape
        nctrg, nytrg, nxtrg = imtrg.shape
        
        if nc!=ncsrc or nysrc!=nytrg or nysrc!=nytrg:
            print('Dimensions of imfold and cmap do not match!')
            return;
        
        ny = nyr*af
        nx = nxr*af2
        
        # Fourier transform in k-space
        src = imsrc
        trg = imtrg
        
        for idx in range(1,3):
            src = fftshift(fft(fftshift(src,axes=idx),axis=idx,norm='ortho'),axes=idx)
            trg = fftshift(fft(fftshift(trg,axes=idx),axis=idx,norm='ortho'),axes=idx)
        
        kernel, sources, targets  = self.calcKernel(src,trg)
        weights = grappa_2d.getWeights(kernel,ny,nx)
        recon   = self.applyWeights(imfold,weights)
        
        n_physcoil = int(np.floor(nc/2))
        recon = np.reshape(recon,(2, n_physcoil, ny, nx),order='F')
        reconSNR = np.zeros((2,ny,nx))
        Rn_inv1 = inv(corr_mtx[0:n_physcoil,0:n_physcoil])
        Rn_inv2 = inv(corr_mtx[n_physcoil:,n_physcoil:])
        
        for ip in range(0,2):
            for iy in range(0,ny):
                for ix in range(0,nx):
                    p = recon[ip,:,iy,ix]
                    p = np.reshape(p,(p.size,1))
                    
                    if ip==0:
                        reconSNR[ip,iy,ix] = np.sqrt(np.abs(p.T @ (Rn_inv1 @ p.conj())))
                    else:
                        reconSNR[ip,iy,ix] = np.sqrt(np.abs(p.T @ (Rn_inv2 @ p.conj())))
        recon = np.sqrt(np.sum(np.abs(recon)**2,axis=1))
        
        gfact = self.calcGfact(weights,imtrg)
        
        return recon, gfact
        #%gfact = permute(gfact,[3 1 2]);
    
    def calcKernel(self,acs_src,acs_trg):
        #np.linalg.lstsq(ktot, distorted_rows, rcond=None)[0]
        srcy = self.srcy
        srcx = self.srcx
        af = self.af
        af2 = self.af2
        
        nc, nyacs, nxacs = acs_src.shape

        src = np.zeros((nc*srcy*srcx,(nyacs-(srcy-1)*af)*(nxacs-(srcx-1)*af2)),dtype=np.complex64)
        trg = np.zeros((nc*af*af2,(nyacs-(srcy-1)*af)*(nxacs-(srcx-1)*af2)),dtype=np.complex64)
        
        cnt = 0
        #how to directly array this?
        for xind in range(0,nxacs-af2*(srcx-1)):
            for yind in range(0,nyacs-(srcy-1)*af):
                src[:,cnt] = np.reshape(acs_src[:,yind:(yind+(srcy)*af):af,xind:(xind+(srcx)*af2):af2],(nc*srcy*srcx,), order='F')
                
                ystart = yind + np.floor((af*(srcy-1)+1)/2) - np.floor(af/2)
                ystop = yind + np.floor((af*(srcy-1)+1)/2) - np.floor(af/2) + af
                tmp1 = slice(int(ystart) , int(ystop))
                
                xstart = xind + np.floor((af2*(srcx-1)+1)/2) - np.floor(af2/2)
                xstop = xind + np.floor((af2*(srcx-1)+1)/2) - np.floor(af2/2) + af2
                tmp2 = slice(int(xstart) , int(xstop))
                
                trg[:,cnt] = np.reshape(acs_trg[:,tmp1,tmp2],(nc*af*af2,),order='F')
                
                cnt += 1
        
        ws = trg @ self.pinv_reg(src,tol=1e-4)
        
        ws_tmp = np.reshape(ws,(nc,af,af2,nc,srcy,srcx),order='F')    
        #flip source points in ky and kx for the convolution
        ws_tmp = np.flip(np.flip(ws_tmp,axis=4),axis=5)                      
        ws_kernel = np.zeros((nc,nc,af*srcy, af2*srcx),dtype=np.complex64)
        
        for kx in range(0,af2):
            for ky in range(0,af):
                ws_kernel[:,:,ky::af, kx::af2] = ws_tmp[:,ky,kx,:,:,:]
        
        return ws_kernel, src, trg

    @staticmethod    
    def getWeights(ws_kernel,ny,nx):
        
        nc, nky, nkx = ws_kernel.shape[1:4]
        
        ws_k = np.zeros((nc,nc,ny,nx),dtype=np.complex64)
        ystart = int(np.ceil((ny-nky)/2))
        ystop = int(np.ceil((ny+nky)/2))
        yrange = slice(ystart,ystop)
        
        xstart = int(np.ceil((nx-nkx)/2))
        xstop = int(np.ceil((nx+nkx)/2))
        xrange = slice(xstart,xstop)
        
        ws_k[:,:,yrange,xrange] = ws_kernel  #put reconstruction kernel in the center of matrix

        tmp0 = ifftshift(ws_k,axes=2)           # shift in phase
        tmp1 = ifftshift(tmp0,axes=3)           # shift in read
        tmp0 = ifft(tmp1,axis=2,norm='ortho')             # ifft in phase
        tmp1 = ifft(tmp0,axis=3,norm='ortho')             # ifft in read
        tmp0 = ifftshift(tmp1,axes=2)           # shift in phase
        tmp1 = ifftshift(tmp0,axes=3)          # shift in read
        
        ws_img = np.sqrt(ny*nx)*tmp1
        return ws_img
    
        
    def applyWeights(self,img_red,ws_img):

        af = self.af
        af2 = self.af2
        
        nc, ny, nx = ws_img.shape[1:]
        nc, nyr, nxr = img_red.shape
        
        if (ny > nyr):
          
            af = round(ny/nyr)
            af2 = round(nx/nxr)
            print('Assuming the data is passed without zeros at not sampled lines ......')
            print('Acceleration factor is af = ' + str(af) + '.....')
            sig_red = img_red
            for idx in range(1,3):
                sig_red = fftshift(fft(fftshift(sig_red,axes=idx),axis=idx,norm='ortho'),axes=idx)
                
            sig_new = np.zeros((nc,ny,nx),dtype=np.complex64)
            sig_new[:,::af,::af2] = sig_red
            img_red = sig_new
            for idx in range(1,3):
                img_red = ifftshift(ifft(ifftshift(img_red,axes=idx),axis=idx,norm='ortho'),axes=idx)
                
        recon = np.zeros((nc,ny,nx),dtype=np.complex64)
        
        for k in range(0,nc):
            recon[k,:,:] = np.sum(ws_img[k,:,:,:] * img_red,axis=0)
            
        return recon
    
    def calcGfact(self,ws_img,imgref):
        af = self.af
        af2 = self.af2
        corr_mtx = self.corr_mtx
        
        nc, ny, nx = ws_img.shape[1:]
        
        ws_img = ws_img/(af*af2)
        
        print('Calculating G-factor......')
        sigref = imgref
        for idx in range(1,3):
            sigref = fftshift(fft(fftshift(sigref,axes=idx),axis=idx,norm='ortho'),axes=idx)
                
        nc, nyref, nxref = sigref.shape
        
        # filter kspace
        sigref = sigref * np.reshape(tukey(nyref,alpha=1),(1, nyref, 1))
        sigref = sigref * np.reshape(tukey(nxref,alpha=1),(1, 1, nxref))
        # 
        sigreff = np.zeros((nc,ny,nx),dtype=np.complex64)
        yidx = slice(int(np.floor((ny-nyref)/2)),int(nyref + np.floor((ny-nyref)/2)))
        xidx = slice(int(np.floor((nx-nxref)/2)),int(nxref + np.floor((nx-nxref)/2)))
        sigreff[:,yidx,xidx] = sigref
        
        imgref = sigreff
        for idx in range(1,3):
            imgref = ifftshift(ifft(ifftshift(imgref,axes=idx),axis=idx,norm='ortho'),axes=idx)
                    
        g = np.zeros((ny,nx))
        
        for y in range(0,ny):
            for x in range(0,nx):
                tmp = imgref[:,y,x]
                norm_tmp = norm2(tmp,2)
                
                W = ws_img[:,:,y,x]            # Weights in image space
                n = tmp.conj().T / norm_tmp
                # This is the generalized g-factor formulation
                g[y,x] = np.sqrt(np.abs((n@W)@(corr_mtx@((n@W).conj().T)))) / np.sqrt(np.abs(n@(corr_mtx@(n.conj().T))))       
       
        return g
    
    #@staticmethod
    def pinv_reg(self,A,tol):
        #PINV   Pseudoinverse.
        #   X = PINV(A) produces a matrix X of the same dimensions
        #   as A' so that A*X*A = A, X*A*X = X and A*X and X*A
        #   are Hermitian. The computation is based on SVD(A) and any
        #   singular values less than a tolerance are treated as zero.
        #   The default tolerance is MAX(SIZE(A)) * NORM(A) * EPS.
        #
        #   PINV(A,TOL) uses the tolerance TOL instead of the default.

        m, n = A.shape
        
        if n > m:
           
           X = self.pinv_reg(A.conj().T,tol=tol).conj().T
        else:
            
           AA = A.conj().T @ A
           Z = np.identity(AA.shape[0])
           S = svd(AA,full_matrices=False,compute_uv=False)
           S = np.sqrt(np.max(np.abs(S)))
           X = lstsq((AA + (Z * tol * S**2)), A.conj().T)[0]

        return X