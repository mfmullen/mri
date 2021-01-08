# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 13:00:30 2020

@author: Mike
"""
import numpy as np
import scipy.sparse as sparse
import scipy.interpolate as interp
from scipy.ndimage import map_coordinates
from scipy.linalg import lstsq, solve

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
            self.num_splines =self.image_size//self.spline_spacing + 1
        except (AttributeError, TypeError, ValueError):
            raise Exception('spline_spacing and image_size must be 3x1 numpy arrays')
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

        tmpx = np.linspace(0,nx-1,num=nx,dtype=np.float32)
        tmpy = np.linspace(0,ny-1,num=ny,dtype=np.float32)
        tmpz = np.linspace(0,nz-1,num=nz,dtype=np.float32)

        x, y, z = np.meshgrid(tmpx,tmpy,tmpz,indexing='ij')
        Tlocalx = x + Tx;
        
        return map_coordinates(Iin,(Tlocalx, y, z), order=1,cval=0)
        
            
    def imagegrad(self,Iin, Tx):
        gradX = np.gradient(Iin,axis=0)
        return self.movepixels_3d(gradX, Tx)
        
    def gradstep(self,I1, I2, Tx, Btotal, Bgrad):
        gradTx = np.gradient(Tx,axis=0)
        
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
        df1 = Btotal @ np.reshape(im1 + im3,(im1.size,1),order='F')
        df2 = Bgrad @ np.reshape(im2 + im4,(im2.size,1),order='F')
        
        return -np.reshape((df1 + df2),(df1.size,1),order='F')
    
    def cost_func(self,I1,I2,Tx):
        gradTx = np.gradient(Tx,axis=0)

        I1mod = (1 - gradTx) * self.movepixels_3d(I1,-Tx)
        I2mod = (1 + gradTx) * self.movepixels_3d(I2,Tx)

        return np.sum(np.abs(I1mod[:] - I2mod[:])**2)
    
    def bsplines(self,axis=0):
        x = np.reshape(np.linspace(1,self.image_size[axis],num=self.image_size[axis],dtype=np.float32),(self.image_size[axis],1))
        knotcenters = np.reshape(np.linspace(1,self.image_size[axis],num=self.num_splines[axis],dtype=np.float32),(self.num_splines[axis],1))
        h = self.spline_spacing[axis].astype(np.float32)
        output = np.zeros((self.num_splines[axis], self.image_size[axis]),dtype=np.float32)
        
        dist = np.abs(knotcenters - x.transpose())/h
        
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
        print(f'iter = {current_iter:d} \t norm(grad) = {np.linalg.norm(grad):.6f} \n')

        nx, ny, nz = Tx.shape
        
        costOut = np.zeros((maxIter,1),dtype=np.float32)
        
        stag_ctr = 0
        costOld = 10**10
        
        while np.linalg.norm(grad[:]) > epsilon and current_iter < maxIter and stag_ctr < stag_lim:
                
            lambdaNew = (1 + np.sqrt(1 + 4*(reg_param**2)))/2
            gamma = (1 - reg_param)/lambdaNew
            
            yNew = x - stepsize * grad
            xNew = (1 - gamma) * yNew + gamma * y
            
            x = xNew
            y = yNew
            reg_param = lambdaNew
            
            Tx = np.reshape((xNew.transpose() @ Btotal),(nx, ny, nz),order='F')
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

            print(f'iter = {current_iter:d} \t norm(grad) = {np.linalg.norm(grad):.6f} \t cost = {cost:.6f} \n')
        
        return Tx, costOut
    
    def estimate_b0(self, I1, I2):
        Bx = sparse.csr_matrix(self.bsplines(axis=0))
        By = sparse.csr_matrix(self.bsplines(axis=1))
        Bz = sparse.csr_matrix(self.bsplines(axis=2))

        Bgx = sparse.csr_matrix(np.gradient(self.bsplines(axis=0),axis=1))

        mx = Bx.shape[0]
        my = By.shape[0]
        mz = Bz.shape[0]

        Btotal = sparse.csr_matrix(sparse.kron(Bz,sparse.kron(By,Bx)))
        Bgrad = sparse.csr_matrix(sparse.kron(Bz,sparse.kron(By,Bgx)))
        
        x0 = np.zeros((mx*my*mz,1), dtype=np.float32)
        Tx = np.zeros(self.image_size, dtype=np.float32)
        
        Tx, cost = self.gradient_method_nesterov(I1,I2,Tx,Btotal,Bgrad, x0)
        return Tx, cost
    
    def run_correction(self, I1, I2, Tx):
        
        nx, ny, nz = I1.shape

        
        Ic = np.zeros((nx,ny,nz), dtype=np.float32)
        x = np.reshape(np.linspace(1,nx,num=nx, dtype=np.float32),(nx,1))

        for idx in range(0,ny*nz):
            iy = int(idx % ny)
            iz = int((idx - iy) / nz)
            
            tmp_distortion = np.reshape(Tx[:,iy,iz],(nx,1))
            
            t = np.reshape(x - tmp_distortion,(1,nx))
            kplus = np.sinc(t - x)
            
            t = np.reshape(x + tmp_distortion,(1,nx))
            kminus = np.sinc(t - x)
            
            ktot = np.vstack((kplus,kminus))
            
            col1 = np.reshape(I1[:,iy,iz],(nx,1))
            col2 = np.reshape(I2[:,iy,iz],(nx,1))

            distorted_rows = np.vstack((col1, col2))
            distorted_rows = np.reshape(distorted_rows,(2*nx,1),order='F')
            
            Ic[:,iy,iz] = np.reshape(lstsq(ktot, distorted_rows)[0],(nx,))
    
        return Ic
        
    def run(self,I1,I2):
                
        Tx, cost = self.estimate_b0(I1, I2)
        Ic = self.run_correction(I1,I2,Tx)
        
        return Ic, Tx

        
        
        