# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 13:00:30 2020

@author: Mike
"""
import numpy as np
#import cupyx.scipy.sparse as sparse
import scipy.sparse as scisparse
from cupyx.scipy.ndimage import map_coordinates
#from scipy.linalg import lstsq
#from cupy.linalg import lstsq
from numpy.linalg import lstsq
import cupy as cp
from cupyx.scipy.sparse import csr_matrix as csr_gpu
from scipy.sparse import csr_matrix as csr_cpu
from joblib import Parallel, delayed
import multiprocessing

#def gradient(inputarray, axis = 0):
#    return cp.diff(inputarray,axis = axis,prepend=0)
    
def run_correction(fplus, fminus, dist):
                
        nx = fplus.size
        
        tmp_distortion = np.reshape(dist,(nx,1))
        
        x = np.reshape(np.linspace(0,nx-1,num=nx),(nx,1)).astype(np.float32)
    
        t = np.reshape(x - tmp_distortion,(1,nx))
        kplus = np.sinc(t - x)
        
        t = np.reshape(x + tmp_distortion,(1,nx))
        kminus = np.sinc(t - x)
        
        ktot = np.vstack((kplus,kminus))
        
        col1 = fplus.reshape((nx,1))
        col2 = fminus.reshape((nx,1))
        
        totalcol = np.vstack((col1,col2))
        
        return lstsq(ktot, totalcol,rcond=None)[0].reshape((nx,)).astype(np.float32)

class distortion_correction:
    '''
    Interface to distortion_correction class. All inputs are optional, but the
    input formats must follow that of the defaults.
    
    Parameters:
    spline_spacing=np.array([3,3,3]), 
    image_size=np.array([128, 128, 128]), 
    max_iterations=400,
    tolerance=6.5e-4, 
    stagnate_limit = 10, 
    stepsize=0.1,
    epsilon=1e-4
                 
    '''
    
    def __init__(self, spline_spacing=np.array([3,3,3]), 
                 image_size=np.array([128, 128, 128]), max_iterations=400,
                 tolerance=6.5e-4, stagnate_limit = 10, stepsize=0.1,
                 epsilon=1e-4):
        
        #spline parameters
        self.spline_spacing = spline_spacing
        self.image_size = image_size
        
        try:
            num_splines = (image_size//spline_spacing) + 1
            self.num_splines = num_splines.astype(np.int)
        except (AttributeError, TypeError, ValueError):
            raise Exception('spline_spacing and image_size must be 3x1 cupy arrays')
        except ZeroDivisionError:            
            raise Exception('spline_spacing must be nonzero')

        #iterative algorithm parameters
        self.max_iterations = max_iterations
        if max_iterations <= 0:
            raise Exception('max_iterations must be >= 1')
        self.tolerance = tolerance
        if tolerance <= 0:
            raise Exception('Non-positive tolerance will never converge')
        self.stagnate_limit = stagnate_limit
        if stagnate_limit <= 0:
            raise Exception('Non-positive stagnate_limit will never converge')
        self.stepsize = stepsize
        if stepsize <= 0:
            raise Exception('Non-positive stagnate_limit will never converge and will probably diverge.')

        self.epsilon = epsilon
    
    def movepixels_3d(self,Iin, Tx):
        '''
        This function will translate the pixels of an image
        according to Tx translation images. 
        
        Inputs;
        Tx: The transformation image, dscribing the
                     (backwards) translation of every pixel in the x direction.
       
        Outputs,
          Iout : The transformed image
        
        '''
        
        nx, ny, nz = Iin.shape

        tmpx = cp.linspace(0,nx-1,num=nx,dtype=np.float32)
        tmpy = cp.linspace(0,ny-1,num=ny,dtype=np.float32)
        tmpz = cp.linspace(0,nz-1,num=nz,dtype=np.float32)

        x, y, z = cp.meshgrid(tmpx,tmpy,tmpz,indexing='ij')
        Tlocalx = cp.expand_dims(x + Tx,axis=0)
        y = cp.expand_dims(y,axis=0)
        z = cp.expand_dims(z,axis=0)
        coordinates = cp.vstack((Tlocalx, y, z))
        
        return map_coordinates(Iin, coordinates, order=1,cval=0)
        
            
    def imagegrad(self,Iin, Tx):
        gradX = cp.gradient(Iin,axis=0)
        return self.movepixels_3d(gradX, Tx)
        
    def gradstep(self,I1, I2, Tx, Btotal, Bgrad):
        gradTx = cp.gradient(Tx,axis=0)
        
        I1mod = self.movepixels_3d(I1,-Tx)
        I2mod = self.movepixels_3d(I2,Tx)

        I1grad = self.imagegrad(I1,-Tx)
        I2grad = self.imagegrad(I2,Tx)
        
        imagediff = (1-gradTx) * I1mod - (1+gradTx) * I2mod
        
        im1 = imagediff * (I1grad * (1-gradTx))
        im2 = imagediff * I1mod
        im3 = imagediff * (I2grad * (1 + gradTx))
        im4 = imagediff * I2mod
        
        #Btotal and Bgrad are entered as sparse arrays
        df1 = Btotal.dot(cp.reshape(im1 + im3,(im1.size,1),order='F'))
        df2 = Bgrad.dot(cp.reshape(im2 + im4,(im2.size,1),order='F'))
        
        return -cp.reshape((df1 + df2),(df1.size,1),order='F')
    
    def cost_func(self,I1,I2,Tx):
        gradTx = cp.gradient(Tx,axis=0)

        I1mod = (1 - gradTx) * self.movepixels_3d(I1,-Tx)
        I2mod = (1 + gradTx) * self.movepixels_3d(I2,Tx)

        return cp.sum(cp.abs(I1mod - I2mod)**2)
    
    def bsplines(self,axis=0):
        '''
        x = cp.reshape(cp.linspace(1,self.image_size[axis],num=self.image_size[axis].item(),dtype=np.float16),(self.image_size[axis].item(),1))
        knotcenters = cp.reshape(cp.linspace(1,self.image_size[axis],num=self.num_splines[axis].item(),dtype=np.float16),(self.num_splines[axis].item(),1))
        h = self.spline_spacing[axis].astype(np.float16)
        output = cp.zeros((self.num_splines[axis].item(), self.image_size[axis].item()),dtype=np.float16)
        
        dist = cp.abs(knotcenters - x.transpose())/h
        
        kk = cp.abs(dist) <= 1
        output[kk] = 2/3 - (1 - dist[kk]/2) * dist[kk]**2
    
        kk = cp.logical_and(cp.abs(dist) > 1, cp.abs(dist) <= 2)
        output[kk] = ((2 - dist[kk])**3)/6
        '''
        
        x = np.reshape(np.linspace(1,self.image_size[axis],num=self.image_size[axis],dtype=np.float16),(self.image_size[axis],1))
        knotcenters = np.reshape(np.linspace(1,self.image_size[axis],num=self.num_splines[axis],dtype=np.float16),(self.num_splines[axis],1))
        h = self.spline_spacing[axis].astype(np.float16)
        output = np.zeros((self.num_splines[axis], self.image_size[axis]),dtype=np.float16)
        
        dist = np.abs(knotcenters - x.transpose())/h
        dist = dist.astype(np.float16)
        
        kk = np.abs(dist) <= 1
        output[kk] = 2/3 - (1 - dist[kk]/2) * dist[kk]**2
    
        kk = np.logical_and(np.abs(dist) > 1, np.abs(dist) <= 2)
        output[kk] = ((2 - dist[kk])**3)/6
        return output
    
    def gradient_method_nesterov(self, I1,I2,Tx,Btotal,Bgrad, x0):
        '''
        ======================================
        INPUT
        ======================================
        I1, I2..  Distorted images
        Tx .....  Current displacement estimate
        Btotal..  Sparse GPU array containing the transform matrix from 
                  spline coefficients to spatial distortion.
        Bgrad...  Sparse GPU array containing the gradient of the transform
                  matrix in distorted dimension
        x0......  Initial guess of B-spline coefficients
        opts ...  Contains stepsize and termination criteria
        
        ======================================
        OUTPUT
        ======================================
        Tx .....  The optimal solution within specified tolerance
        cost ..   The value of the objective function at xopt
        '''
        maxIter = self.max_iterations
        epsilon = self.epsilon
        stepsize = self.stepsize
        stag_lim = self.stagnate_limit
        stag_tol = self.tolerance
        
        x = x0
        y = x0
        current_iter = 0
        
        reg_param = 0.0
        
        grad = self.gradstep(I1,I2,Tx,Btotal,Bgrad)
        print(f'iter = {current_iter:d} \t norm(grad) = {cp.linalg.norm(grad).get():.6f} \n')

        nx, ny, nz = Tx.shape
        
        costOut = cp.zeros((maxIter,1),dtype=np.float32)
        
        stag_ctr = 0
        costOld = 10**10
        
        while cp.linalg.norm(grad.ravel()) > epsilon and current_iter < maxIter and stag_ctr < stag_lim:
                
            lambdaNew = (1 + cp.sqrt(1 + 4*(reg_param**2)))/2
            gamma = (1 - reg_param)/lambdaNew
            
            yNew = x - stepsize * grad
            xNew = (1 - gamma) * yNew + gamma * y
            
            x = xNew
            y = yNew
            reg_param = lambdaNew
            
            #Tx = cp.reshape(cp.dot(xNew.transpose(), Btotal),(nx, ny, nz),order='F')
            Tx = cp.reshape(Btotal.T.dot(xNew),(nx, ny, nz),order='F')

            grad = self.gradstep(I1,I2,Tx,Btotal,Bgrad)
            cost = self.cost_func(I1,I2,Tx)
            
            #when on gpu, will need to gather cost
            costOut[current_iter] = cost;
            
#            if cost > costOld:
  #              break
            
            if (costOld - cost)/costOld < stag_tol:
                stag_ctr += 1
            else:
                stag_ctr = 0
            
            costOld = cost#costOut[current_iter]
            current_iter += 1

            print(f'iter = {current_iter:d} \t norm(grad) = {cp.linalg.norm(grad).get():.6f} \t cost = {cost.get():.6f} \n')
        
        return Tx, costOut
    
    def estimate_b0(self, I1, I2):
        nx, ny, nz = I1.shape
        
        Bx = csr_cpu(self.bsplines(axis=0))
        By = csr_cpu(self.bsplines(axis=1))
        Bz = csr_cpu(self.bsplines(axis=2))

        Bgx = csr_cpu(np.gradient(self.bsplines(axis=0),axis=1))

        mx = Bx.shape[0]
        my = By.shape[0]
        mz = Bz.shape[0]

        Btotal = csr_cpu(scisparse.kron(Bz,scisparse.kron(By,Bx)))
        Bgrad = csr_cpu(scisparse.kron(Bz,scisparse.kron(By,Bgx)))
        
        x0 = cp.zeros((mx*my*mz,1), dtype=np.float32)
        Tx = cp.zeros((nx, ny, nz), dtype=np.float32)
        cost = 0
        Tx, cost = self.gradient_method_nesterov(I1,I2,Tx,csr_gpu(Btotal),csr_gpu(Bgrad), x0)
        return Tx, cost
        
        
    def run(self,I1,I2):
                
        Tx, cost = self.estimate_b0(cp.asarray(I1), cp.asarray(I2))
        Tx = Tx.get()
        cp._default_memory_pool.free_all_blocks()
        
        #need to define the linspace below outside of run_correction because
        #numba does not have a very flexible interface for linspace
        nx, ny, nz = I1.shape
        num_cores = int(multiprocessing.cpu_count()*.8)
        I1 = np.reshape(I1,(nx,ny*nz))
        I2 = np.reshape(I2,(nx,ny*nz))
        Tx = np.reshape(Tx,(nx,ny*nz))
        
        Ic = Parallel(n_jobs=num_cores)(delayed(run_correction)(I1[:,idx],I2[:,idx],Tx[:,idx]) for idx in range(0,ny*nz))
        Ic = np.reshape(np.asarray(Ic).T,(nx,ny,nz)) 
        Tx = np.reshape(Tx,(nx,ny,nz))
        
        return Ic, Tx

        
        
        