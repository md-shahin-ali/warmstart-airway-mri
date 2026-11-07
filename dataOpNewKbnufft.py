#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===============================================================
#  dataOpNewKbnufft.py  --  data loading and forward operators
#
#  Reads one slice of spiral k-space from a MATLAB v7.3 .mat file, performs
#  coil compression to virtual coils, estimates coil sensitivities
#  (ESPIRiT by default), splits each readout into disjoint training /
#  validation samples for the self-supervised loss, and precomputes A^H b
#  and Toeplitz NUFFT kernels.
#
#  Expected .mat contents:
#      kdata : complex [nReadouts, nCoils, nArms, nSlices]
#      k     : complex [nReadouts, nArms]   (kx + i*ky, normalized to [-0.5, 0.5])
#      dcf   : real    [nReadouts, nArms]
#
#  The first time a slice is read it is cached next to the .mat file as
#  <stem>_<zero-based slice>.pickle; later runs read the cache instead.
# ===============================================================

from os import path
import numpy as np
import torch
import torchkbnufft as tkbn

from espirit.espirit import espirit, fft
import pickle
import mat73
import tqdm
import matplotlib.pyplot as plt
import scipy as scipy
import sigpy as sp
import sigpy.mri as mr
import sigpy.plot as pl
import scipy.stats as stats
from scipy.stats import truncnorm

# ismrmrdtools is only needed for coilEst='inati' or 'walsh'; the default
# ('espirit') runs without it.
try:
    from ismrmrdtools import coils, show
except ImportError:
    coils = show = None

import numpy as np
from sklearn.decomposition import PCA

def coil_compress_withpca(kSpace, no_comp):
    """
    Function to perform coil compression using PCA.

    Parameters:
    kSpace (numpy.ndarray): Input k-space data with shape (Readouts, Rays, Coils).
    no_comp (int): Number of principal components after compression.

    Returns:
    numpy.ndarray: Compressed k-space data.
    """
    # Convert to double precision and remove singleton dimensions
    data = np.squeeze(kSpace).astype(np.float64)
    nreadouts, ntotalrays, nch = data.shape

    # Reshape data for PCA
    data = data.reshape(nreadouts * ntotalrays, nch)

    # Perform PCA
    pca = PCA(n_components=no_comp)
    compressed_data = pca.fit_transform(data)

    # Reshape compressed data back to original dimensions
    compressed_data = compressed_data.reshape(nreadouts, ntotalrays, no_comp)

    return compressed_data


def skewed_sample_with_indices(array, percentage=5, skewness=0, mean=1.5):
    """
    Generates a sample from the array following a right-skewed normal distribution.

    :param array: 1-D array to sample from.
    :param percentage: Percentage of elements to sample (default is 5%).
    :param skewness: Skewness of the distribution (default is 5, higher values mean more right-skewed).
    :return: A tuple containing the sample of the array and the indices of the sampled elements.
    """
    sample_size = int(len(array) * (percentage / 100))
    norm_dist = stats.norm(loc=mean, scale=2)
    probabilities = norm_dist.pdf(np.linspace(-3, 3, len(array)))
    skewed_probabilities = np.power(probabilities, skewness)
    skewed_probabilities /= skewed_probabilities.sum()
    sample_indices = np.random.choice(len(array), size=sample_size, replace=False, p=skewed_probabilities)
    return sample_indices

def uniform_sample_with_indices(array, percentage=5):
    """
    Generates a sample from the array following a uniform distribution.

    :param array: 1-D array to sample from.
    :param percentage: Percentage of elements to sample (default is 5%).
    :return: A tuple containing the sample of the array and the indices of the sampled elements.
    """
    sample_size = int(len(array) * (percentage / 100))
    sample_indices = np.random.choice(len(array), size=sample_size, replace=False)
    return sample_indices


#%% Reads the data and operators
class dataAndOperators:

  # Initialization  
  #---------------------------
    def __init__(self,params):

        self.params = params
        self.im_size = params["im_size"]
        self.batch_size = params["nBatch"]
        dtype = params['dtype']
        gpu=torch.device(params['device'])
        self.gpu = gpu

        if(self.params["verbose"]):
            print("Reading data ..")
    
        # Reading data from mat file
        #----------------------------------------------
        if(params['filename'][-3:-1] =='ma'):  # mat file
            extension = '_'+str(params['slice'])+'.pickle'
            fnamepickle = params['filename'].replace('.mat',extension)    
            if(not(path.exists(fnamepickle))):
                data_dict = mat73.loadmat(params['filename'])
                kdata = data_dict['kdata']
                kdata = np.squeeze(kdata[:,:,:,params['slice']])
                ktraj=np.asarray(data_dict['k'])    
                dcf=np.asarray(data_dict['dcf'])

            # save with pickle for fast reading
                with open(fnamepickle, 'wb') as f:
                    pickle.dump([kdata,ktraj,dcf],f,protocol=4)
            else:
                with open(fnamepickle, 'rb') as f:
                    [kdata,ktraj,dcf] = pickle.load(f)
        
        else: # read pickle file
            fname = params['filename']  
            with open(fname, 'rb') as f:
                [kdata,ktraj,dcf] = pickle.load(f)
    
        #Reshaping the variables
        #----------------------------------------------
        """
            Input Data Shape: kdata [ nReadouts, nCoils, nArms, nSlice ]
                              ktraj [ nReadouts, nArms ]
                              dcf [ nReadouts, nArms ]
        """
        kdata = kdata.astype(np.complex64)
        ktraj=ktraj.astype(np.complex64)
        #kspnorm = torch.zeros(kdata.shape,dtype=torch.complex64)
        #for i in range(kdata.shape[0]):
       #     maxvalue = torch.view_as_real(kdata[i,...]).max()
       #     kspnorm[i,...] = kdata[i,...]/maxvalue
       #     del maxvalue
       # self.kdata = kspnorm
       # del kspnorm
        # normalize kspace done
        if isinstance(params['slice'], int):
            kdata = np.expand_dims(kdata,3)
            
        """
        Reshape data into: kdata [ nCoils, nArms, nReadouts, nSlice ]
                            ktraj [ nArms, nReadouts ]
                            dcf [ nArms, nReadouts ]
        """
        
        kdata=np.transpose(kdata,(1,2,0,3)) 
        dcf = np.transpose(dcf,(1,0))
        ktraj = np.transpose(ktraj,(1,0))

        # Reducing the image size if factor < 1
        #----------------------------------------------

        im_size = np.int_(np.divide(params["im_size"],params["factor"]))
        ktraj=np.squeeze(ktraj)*2*np.pi

        # Deleting initial interleaves to achieve steady state
        #------------------------------------------------------
        self.nch = np.size(kdata,0)
        self.nkpts = np.size(kdata,2)

        nintlvsNeeded = params["nintlPerFrame"]*params["nFramesDesired"]
        nintlvs = np.size(kdata,1)
        nintlvsLeft = nintlvs - params["nintlvsToDelete"]
        if(nintlvsNeeded > nintlvsLeft):
            print("Too few interleaves. Reduce nFramesDesired or nintlvsToDelete")
    
        self.nsl = kdata.shape[3]



        # Reconstructing coil images 
        #---------------------------
    
        if(self.params["verbose"]):
            print("Reconstruction of coil images ..")
        
        self.nch = kdata.shape[0]
        print(kdata.shape)

        kdata = np.reshape(kdata[:,params["nintlvsToDelete"]:nintlvsNeeded+params["nintlvsToDelete"]],(self.nch,nintlvsNeeded*self.nkpts,self.nsl)) 

        ktraj=ktraj[params["nintlvsToDelete"]:nintlvsNeeded+params["nintlvsToDelete"],:]
        dcf = dcf[params["nintlvsToDelete"]:nintlvsNeeded+params["nintlvsToDelete"],:]
        dcf = dcf/params["nintlPerFrame"]/params["nintlPerFrame"]

        ktraj = np.reshape(ktraj,(1,nintlvsNeeded*self.nkpts))
        ktraj = np.stack((np.real(ktraj), np.imag(ktraj)),axis=1)
        dcf = np.reshape(dcf,(1,nintlvsNeeded*self.nkpts)) 
        
        for i in range(self.nsl):
            for j in range(self.nch):
                kdata[j,:,i] = kdata[j,:,i] * dcf

        self.kdata = torch.tensor(kdata,dtype=torch.complex64).unsqueeze(0)
        
        self.ktraj = torch.tensor(ktraj,dtype=torch.float)
        self.dcf = torch.tensor(dcf,dtype=torch.float)

        self.adjnufft_ob=tkbn.KbNufftAdjoint(im_size=im_size,grid_size=im_size,device=gpu)
        coilimages = np.zeros((self.nch,im_size[0],im_size[1],self.nsl)).astype(complex)
    
        for i in range(self.nsl):
            coilimages[...,i] = self.adjnufft_ob(self.kdata[...,i].to(gpu),self.ktraj.to(gpu)).squeeze(0).cpu()
        
        self.kdata = self.kdata.squeeze(3)
        #np.save('Coilimage.npy', coilimages)
        # FOIVR coil combination
        #------------------------------------------------------
    
        if(self.params["verbose"]):
            print("Coil combination ..")
        
        nCoils = params["virtual_coils"]
        x = np.arange(im_size[0])-im_size[0]/2
        y = np.arange(im_size[1])-im_size[1]/2
        x,y = np.meshgrid(x, y)
        mask = x**2 + y**2 < params["mask_size"]*im_size[0]*im_size[1]/4

        signal = coilimages*mask[None,:,:,None]
        noise = coilimages*np.logical_not(mask[None,:,:,None])
            
        signal = np.reshape(signal,(self.nch,im_size[0]*im_size[1]*self.nsl))
        noise = np.reshape(noise,(self.nch,im_size[0]*im_size[1]*self.nsl))
        print(f'Signal shape: {signal.shape}')
        print(f'Noise shape: {noise.shape}')
        
        A = np.real(signal@np.transpose(np.conj(signal)))
        B = np.real(noise@np.transpose(np.conj(noise)))
        [D,W] = scipy.linalg.eig(A,B)
        ind=np.flipud(np.argsort(D))
        W=W[:,ind[0:nCoils]]
        print(f'W is of shape: {W.shape}')

        coilimages = W.T@np.reshape(coilimages,(self.nch,im_size[0]*im_size[1]*self.nsl))
        coilimages = np.reshape(coilimages,(nCoils,im_size[0],im_size[1],self.nsl))
        coilimages = np.expand_dims(coilimages,0)

        kdata = W.T@np.reshape(kdata,(self.nch,params["nFramesDesired"]*params["nintlPerFrame"]*self.nkpts*self.nsl))
        self.kdata = torch.tensor(kdata,dtype=torch.complex64).unsqueeze(0)
        self.nch = nCoils
        self.coilimages = coilimages

    # Coil sensitivity estimation
    #-----------------------------               
            
        if(self.params["coilEst"]=='espirit'):
            if(self.params["verbose"]):
                print("Coil sensitivity estimation using Espirit ..")
            
            if(self.params["CoilMapPrecomputed"]== True):
                # smap = np.load('ESPIRIT_def.npy', allow_pickle=True)
                
                #----External coil maps (path set by params['coil_map_path'])------------------------------------
                smap = np.load(self.params.get('coil_map_path', 'ESPIRIT_def.npy'), allow_pickle=True)
                smap = np.squeeze(smap)
                #--------------------------------------------------------------------------------------------------
            
                print("preComputed Espirit Coil sensitivity mapsloaded!")
                pl.ImagePlot(np.squeeze(smap[:,:,:,2]),z=0,mode='m',colormap='jet',title='Log magnitude of ESPIRIT maps')
                
            else:
                x_f = fft(coilimages, (2, 3))
                #print(x_f.shape)
                smap=np.zeros((im_size[0],im_size[1],self.nch,self.nsl)).astype(complex)

                for nslc in range(self.nsl):
                    x_f1=x_f[:,:,:,:,nslc]
                    x_f1 = np.transpose(x_f1, (2, 3, 0, 1))  # 6, 24
                    smap1 = espirit(x_f1, 6, 24, 0.02, 0.95)#0.14, 0.8925)
                    smap1 = smap1[:, :, 0, :, 0]
                    smap[:,:,:,nslc]=smap1
                np.save('ESPIRIT_def.npy',smap)  
                mp_s= np.absolute(np.transpose(smap,(2,0,1,3)))
                pl.ImagePlot(np.squeeze(mp_s[:,:,:,0]),z=0,mode='m',colormap='jet',title='Log magnitude of ESPIRIT maps')
                print("Coil sensitivity estimation using Espirit Done!")
                smap = np.transpose(smap, (2, 0, 1,3))  # 6, 24
                del mp_s, x_f1, coilimages
                
            self.smap=np.expand_dims(smap,axis=0)

        elif (self.params["coilEst"]=='jsense'):
            if(self.params["verbose"]):
                print("Coil sensitivity estimation using Jsense ..")
            smap = self.giveJsenseCoilsensitivities(coilimages.squeeze(0),0.03)
            mp_s= np.absolute(smap)
            pl.ImagePlot(np.squeeze(mp_s[:,:,:,0]),z=0,mode='m',colormap='jet',title='Log magnitude of JSENSE maps')
            print("Coil sensitivity estimation using Jsense Done!")
            del mp_s
            #print(smap.shape) ##(8, 168, 168, 4)
            np.save('JSENSE_def.npy',smap) 
            self.smap=np.expand_dims(smap,axis=0)
        
        elif (self.params["coilEst"] in ('inati', 'walsh')) and coils is None:
            raise ImportError("coilEst='%s' needs the ismrmrdtools folder (included "
                              "in this repository) next to this file"
                              % self.params["coilEst"])

        elif (self.params["coilEst"]=='inati'):
            # coilimages: [1, nCoils, x, y, nSlc]
            smap=np.zeros((self.nch, im_size[0],im_size[1],self.nsl)).astype(complex)
            for slc in range(self.nsl):
                (smap[:,:,:,slc], CoilCombined)= coils.calculate_csm_inati_iter(np.squeeze(self.coilimages[0,:,:,:,slc]), smoothing=7, niter=10, verbose=True)
                show.imshow(abs(smap[...,slc]), tile_shape=(int(self.nch/2), 2), scale=(0, 1))
                show.imshow(abs(CoilCombined), cmap='gray',colorbar=True, titles=['coilCombined by inati smaps'])
                
            self.smap=np.expand_dims(smap,axis=0)
            
        elif (self.params["coilEst"]=='walsh'):
            # coilimages: [1, nCoils, x, y, nSlc]
            smap=np.zeros((self.nch, im_size[0],im_size[1],self.nsl)).astype(complex)
            for slc in range(self.nsl):
                (smap[:,:,:,slc], CoilCombined)= coils.calculate_csm_walsh(np.squeeze(self.coilimages[0,:,:,:,slc]), smoothing=30, niter=10)
                show.imshow(abs(smap[...,slc]), tile_shape=(int(self.nch/2), 2), scale=(0, 1))
                show.imshow(abs(CoilCombined), cmap='gray',colorbar=True, titles=['coilCombined by walsh smaps'])
                
            self.smap=np.expand_dims(smap,axis=0)

        self.toep_ob = tkbn.ToepNufft().to(gpu)
    
        self.changeNumFrames(params)
        
        # function to change the number of frames
    def changeNumFrames(self,params):
        gpu = self.gpu
        self.smapT = torch.tensor(self.smap,dtype=torch.complex64)
        self.smapT = torch.tile(self.smapT,[params["nBatch"],1,1,1,1])

        ktrajSorted = torch.reshape(self.ktraj,(1,2,params["nFramesDesired"], params["nintlPerFrame"]*self.nkpts))  
        ktrajSorted = ktrajSorted.permute(2,1,3,0).squeeze().contiguous()

        kdataSorted = torch.reshape(self.kdata,(1,self.nch,params["nFramesDesired"],  params["nintlPerFrame"]*self.nkpts,self.nsl))
        kdataSorted = kdataSorted.permute(2,1,3,0,4).squeeze(3).contiguous()
        
        dcfSorted = torch.reshape(self.dcf,(1,params["nFramesDesired"],params["nintlPerFrame"]*self.nkpts))
        dcfSorted = dcfSorted.permute(1,0,2).contiguous()
        
        """
        Current data shape: 
            (dcfCompensated)kdataSorted :[ nFramesDesired, nCoils, params["nintlPerFrame"]*nReadouts, nSlice ]              
                             ktrajSorted:[ nFramesDesired, 2, params["nintlPerFrame"]*nReadouts ]
                               dcfSorted:[ nFramesDesired, 1, params["nintlPerFrame"]*nReadouts ]
        """
        
        # Disjoint splitting
    #-------------------
    
        tempKdata = kdataSorted.view(self.params["nFramesDesired"], self.nch,self.params["nintlPerFrame"], self.nkpts, self.nsl)
        tempKtraj = ktrajSorted.view(self.params["nFramesDesired"], 2, self.params["nintlPerFrame"], self.nkpts)
        tempDcf = dcfSorted.view(self.params["nFramesDesired"], 1, self.params["nintlPerFrame"], self.nkpts)
        
        tempKdata_usp = torch.zeros((self.params["nFramesDesired"], self.nch, self.params['Usp_arms_PerFrame'], self.nkpts, self.nsl), dtype= torch.complex64)
        tempKtraj_usp = torch.zeros((self.params["nFramesDesired"], 2, self.params['Usp_arms_PerFrame'], self.nkpts), dtype= torch.float)
        tempDcf_usp = torch.zeros((self.params["nFramesDesired"], 1, self.params['Usp_arms_PerFrame'], self.nkpts), dtype= torch.float)
        
        if self.params['Undersample']:
            print(f' Retrospective undersampling is done within each frame')
            if self.params['Usp_arms_PerFrame']==3:
                tempKdata_usp = torch.zeros((self.params["nFramesDesired"], self.nch, self.params['Usp_arms_PerFrame'], self.nkpts, self.nsl), dtype= torch.complex64)
                tempKtraj_usp = torch.zeros((self.params["nFramesDesired"], 2, self.params['Usp_arms_PerFrame'], self.nkpts), dtype= torch.float)
                tempDcf_usp = torch.zeros((self.params["nFramesDesired"], 1, self.params['Usp_arms_PerFrame'], self.nkpts), dtype= torch.float)
                locs= [0,1,2] #[0, 12, 27 ] #[0, 18]
                
                for frame in range(self.params["nFramesDesired"]):
                    print(f'picking {locs}-th arms from {frame}-th frame')
                    tempKdata_usp[frame]= tempKdata[frame, :, locs, :, :]
                    tempKtraj_usp[frame]= tempKtraj[frame,:,locs, :]
                    tempDcf_usp[frame]= tempDcf[frame,:,locs, :]
                    for i in range(len(locs)):
                        locs[i]+=3
                        if locs[i]>34:
                            locs[i]=locs[i]%35
                
                self.params["nintlPerFrame"]= self.params['Usp_arms_PerFrame']
                
                tempKdata = tempKdata_usp
                tempKtraj = tempKtraj_usp
                tempDcf = tempDcf_usp
                
                del tempKdata_usp, tempKtraj_usp, tempDcf_usp
                
                print(f'updated nintlPerFrame: {self.params["nintlPerFrame"]}')
                print('input data shapes after retrospective undersampling')
                print(f'kdata_usp:{tempKdata.shape}; {tempKdata.dtype}')
                print(f'ktraj_usp:{tempKtraj.shape}; {tempKtraj.dtype}')
                print(f'dcf_usp:{tempDcf.shape}; {tempDcf.dtype}')
        
        splitRatio= self.params['splitRatio']
        nReadout = self.nkpts
        # Define the number of indices to pick
        num_val_indices = int(nReadout*splitRatio)
        num_trn_indices = nReadout - num_val_indices
        
        kdataTrn = torch.zeros((self.params["nFramesDesired"], self.nch,self.params["nintlPerFrame"],num_trn_indices, self.nsl), dtype= torch.complex64)
        kdataVal = torch.zeros((self.params["nFramesDesired"], self.nch,self.params["nintlPerFrame"],num_val_indices, self.nsl), dtype= torch.complex64)
        ktrajTrn = torch.zeros((self.params["nFramesDesired"], 2, self.params["nintlPerFrame"], num_trn_indices), dtype= torch.float)
        ktrajVal = torch.zeros((self.params["nFramesDesired"], 2, self.params["nintlPerFrame"], num_val_indices), dtype= torch.float)
        dcfTrn = torch.zeros((self.params["nFramesDesired"], 1, self.params["nintlPerFrame"], num_trn_indices), dtype= torch.float)
        dcfVal = torch.zeros((self.params["nFramesDesired"], 1, self.params["nintlPerFrame"], num_val_indices), dtype= torch.float)
        
        
        for frame in range(self.params["nFramesDesired"]):
            for arm in range(self.params["nintlPerFrame"]):
                
                # Create a list of indices
                indices = np.arange(nReadout)
                
                if self.params['splitDist']=='rightSkewed':
                    picked_indices = set(skewed_sample_with_indices(indices, percentage=splitRatio*100, skewness=5, mean=1.5))
                    
                elif self.params['splitDist']=='leftSkewed':
                    picked_indices = set(skewed_sample_with_indices(indices, percentage=splitRatio*100, skewness=5, mean=-1.5))
                    
                elif self.params['splitDist']=='normalDist':
                    picked_indices = set(skewed_sample_with_indices(indices, percentage=splitRatio*100, skewness=0, mean=0))
                    
                elif self.params['splitDist']=='uniformDist':
                    picked_indices = set(uniform_sample_with_indices(indices, percentage=splitRatio*100))
                # New option for random distribution
                elif self.params['splitDist'] == 'randomDist':
                    num_val_indices = int(nReadout * splitRatio)  # number of validation indices
                    picked_indices = set(np.random.choice(indices, num_val_indices, replace=False))  # Random selection
                
                # Create another list that contains the rest of the indices
                rest_indices = list(set(indices) - picked_indices)
                # Convert the set back to a list
                picked_indices = list(picked_indices)

                valIndices = sorted(picked_indices)
                trainIndices = sorted(rest_indices)
                
                kdataTrn[frame,:,arm,:,:] = tempKdata[frame, :, arm, trainIndices, :]
                kdataVal[frame,:,arm,:,:] = tempKdata[frame, :, arm, valIndices, :]
                ktrajTrn[frame,:, arm,:] = tempKtraj [frame, :, arm, trainIndices]
                ktrajVal[frame,:, arm,:] = tempKtraj [frame, :, arm, valIndices]
                dcfTrn [frame, 0, arm, :] = tempDcf [frame, 0, arm, trainIndices]
                dcfVal [frame, 0, arm, :] = tempDcf [frame, 0, arm, valIndices]
                
        dist= self.params['splitDist']
        print(f'{dist} normal dist is applied and {len(picked_indices)} samples are picked \n\n')
        
        # kdata splitting and reshaping
        kdataSortedTrn = torch.reshape(kdataTrn, (self.params["nFramesDesired"], self.nch, self.params["nintlPerFrame"]*kdataTrn.shape[3], self.nsl))
        kdataSortedVal = torch.reshape(kdataVal, (self.params["nFramesDesired"], self.nch, self.params["nintlPerFrame"]*kdataVal.shape[3], self.nsl))
        
#         # ktraj splitting and reshaping
        ktrajSortedTrn = torch.reshape(ktrajTrn, (self.params["nFramesDesired"], 2, self.params["nintlPerFrame"]*ktrajTrn.shape[3]))
        ktrajSortedVal = torch.reshape(ktrajVal, (self.params["nFramesDesired"], 2, self.params["nintlPerFrame"]*ktrajVal.shape[3]))
        
#         # dcf splitting and reshaping
        dcfSortedTrn = torch.reshape(dcfTrn, (self.params["nFramesDesired"], 1, self.params["nintlPerFrame"]*dcfTrn.shape[3]))
        dcfSortedVal = torch.reshape(dcfVal, (self.params["nFramesDesired"], 1, self.params["nintlPerFrame"]*dcfVal.shape[3]))

        # ---------------------------------------------------------------
        # ATTEMPT 4: keep the TRAINING split available to reference_builder.
        #
        # The self-supervised validation loss holds out a fraction of readout
        # samples. If the 35-arm reference image were built from the full k-space
        # it would contain some of those held-out samples, so the network would see
        # validation data through its own input and the validation loss would stop
        # being a clean measure of generalisation.
        #
        # These arrays are already computed above and were otherwise discarded, so
        # this costs nothing but the memory to hold them. reference_builder uses
        # them when present and falls back to the full k-space with a warning when
        # absent.
        # ---------------------------------------------------------------
        self.kdataSortedTrn = kdataSortedTrn
        self.ktrajSortedTrn = ktrajSortedTrn
        
    #----------------------------------------------------------------------------
        
        sz = (params["nFramesDesired"],1,self.im_size[0],self.im_size[1],self.nsl)
    
        if(self.params["verbose"]):
            print("Precomputing Atb ..")
    
       
        if(self.params["fastMode"]):
            self.Atb = torch.zeros(sz,dtype=torch.complex64)
        #-----------------------------------------------------
            # Placeholder for Train and Val Atb
            self.AtbTrn = torch.zeros(sz,dtype=torch.complex64)
            self.AtbVal = torch.zeros(sz,dtype=torch.complex64)
        #-----------------------------------------------------
            for i in range(self.nsl):
                kdata1 = kdataSorted[:,:,:,i].to(gpu)
                ktraj1 = ktrajSorted.to(gpu)
                temp3 = torch.tensor(self.smap[...,i],dtype=torch.complex64)
                smap1 = torch.tile(temp3,[params["nFramesDesired"],1,1,1]).to(gpu)
                self.Atb[...,i] = self.adjnufft_ob(kdata1,ktraj1,smaps=smap1).cpu() # adjoin_kdata1 shape: (batch,coil, klength)
                
            #--------------------------------------------------------------------
                # Compute Train Atb
                kdata1 = kdataSortedTrn[:,:,:,i].to(gpu)
                ktraj1 = ktrajSortedTrn.to(gpu)
                temp3 = torch.tensor(self.smap[...,i],dtype=torch.complex64)
                smap1 = torch.tile(temp3,[params["nFramesDesired"],1,1,1]).to(gpu)
                self.AtbTrn[...,i] = self.adjnufft_ob(kdata1,ktraj1,smaps=smap1).cpu() # adjoin_kdata1 shape: (batch,coil, klength)
                
                # Compute Val Atb
                kdata1 = kdataSortedVal[:,:,:,i].to(gpu)
                ktraj1 = ktrajSortedVal.to(gpu)
                temp3 = torch.tensor(self.smap[...,i],dtype=torch.complex64)
                smap1 = torch.tile(temp3,[params["nFramesDesired"],1,1,1]).to(gpu)
                self.AtbVal[...,i] = self.adjnufft_ob(kdata1,ktraj1,smaps=smap1).cpu() # adjoin_kdata1 shape: (batch,coil, klength)
            #--------------------------------------------------------------------
                
            del kdata1, ktraj1, temp3, smap1  
        else:
            self.Atb = torch.zeros(sz,dtype=torch.complex64)
            for i in range(self.nsl):
                for j in range(params["nFramesDesired"]):
                    temp1 = kdataSorted[j:j+1,:,:,i].to(gpu) 
                    temp2 = ktrajSorted[j:j+1].to(gpu)
                    temp3 = self.smapT[0:1,:,:,:,i].to(gpu)
                    self.Atb[j:j+1,0,:,:,i]=self.adjnufft_ob(temp1,temp2,smaps=temp3).cpu() 
            del temp1, temp2, temp3   
    

        if(self.params["verbose"]):
            print("Precomputing Toeplitz kernels ..")
    
    
        # dcfSorted = torch.reshape(self.dcf,(1,params["nFramesDesired"],params["nintlPerFrame"]*self.nkpts))
        # dcfSorted = dcfSorted.permute(1,0,2).contiguous()
        
        sz = (params["nFramesDesired"],2*self.im_size[0],2*self.im_size[1])
        self.dcomp_kernel = torch.zeros(sz,dtype=torch.complex64)
        
    #----------------------------------------------------------------
        # Placeholder for dcomp_kernel during training and validation
        self.dcomp_kernel_trn = torch.zeros(sz,dtype=torch.complex64)
        self.dcomp_kernel_val = torch.zeros(sz,dtype=torch.complex64)
    #----------------------------------------------------------------
    
        for i in range(params["nFramesDesired"]):
            self.dcomp_kernel[i] = tkbn.calc_toeplitz_kernel(ktrajSorted[i].to(gpu),tuple(self.im_size),dcfSorted[i].to(gpu)).cpu()
        #-------------------------------------------------------------------------------------------------------------------------------------------------
            self.dcomp_kernel_trn[i] = tkbn.calc_toeplitz_kernel(ktrajSortedTrn[i].to(gpu),tuple(self.im_size),dcfSortedTrn[i].to(gpu)).cpu()
            self.dcomp_kernel_val[i] = tkbn.calc_toeplitz_kernel(ktrajSortedVal[i].to(gpu),tuple(self.im_size),dcfSortedVal[i].to(gpu)).cpu()
        #-------------------------------------------------------------------------------------------------------------------------------------------------
            
    
        self.dcomp_kernel = self.dcomp_kernel.unsqueeze(1)
    #-------------------------------------------------------------
        self.dcomp_kernel_trn = self.dcomp_kernel_trn.unsqueeze(1)
        self.dcomp_kernel_val = self.dcomp_kernel_val.unsqueeze(1)
    #-------------------------------------------------------------

        maxvalueAtb = torch.view_as_real(self.Atb).max()
        self.Atb = self.Atb/maxvalueAtb/2
        temp1 = self.Atb[0:1,...,0].to(gpu)
        temp2 = self.dcomp_kernel[0:1].to(gpu)
        temp3 = self.smapT[0:1,:,:,:,0].to(gpu)  
        temp = self.toep_ob(temp1,temp2,smaps=temp3.to(gpu)).cpu()
        maxvalueDcomp = torch.view_as_real(temp).max()
        self.dcomp_kernel = self.dcomp_kernel/maxvalueDcomp
        
        del temp1, temp2, temp3
        
    #-------------------------------------------------------------------
        # Normalize Atb and dcomp_kernel for train and validation
        self.AtbTrn = self.AtbTrn/maxvalueAtb/2
        self.dcomp_kernel_trn = self.dcomp_kernel_trn/maxvalueDcomp

        self.AtbVal = self.AtbVal/maxvalueAtb/2
        self.dcomp_kernel_val = self.dcomp_kernel_val/maxvalueDcomp
    #-------------------------------------------------------------------
    
        self.mask = self.Atb[0:self.batch_size].abs() == 0.00
        
    #------------------------------------------------------------------
        self.maskTrn = self.AtbTrn[0:self.batch_size].abs() == 0.00
        self.maskVal = self.AtbVal[0:self.batch_size].abs() == 0.00
    #------------------------------------------------------------------
        if self.params["fastMode"]:
            print("Moving to GPU")
            # self.Atb = self.Atb.to(gpu)
            # self.dcomp_kernel = self.dcomp_kernel.to(gpu)
            
        #-----------------------------------------------------
            self.AtbTrn = self.AtbTrn.to(gpu)
            self.dcomp_kernel_trn = self.dcomp_kernel_trn.to(gpu)
            
            self.AtbVal = self.AtbVal.to(gpu)
            self.dcomp_kernel_val = self.dcomp_kernel_val.to(gpu)
        #-----------------------------------------------------
            self.smapT = self.smapT.to(gpu)
            #print(self.smapT.shape)
            #print(self.Atb.shape)
        torch.cuda.empty_cache()
    
        if self.params["verbose"]:
            print("Done initializing data Object !!")
            max_memory = torch.cuda.max_memory_allocated(device=gpu)*1e-9
            print("Max GPU utilization",max_memory," GB ")
            current_memory = torch.cuda.memory_allocated(device=gpu)*1e-9
            print("Current GPU utilization",current_memory," GB ")
            available_memory = torch.cuda.get_device_properties(device=gpu).total_memory*1e-9
            print("Total GPU memory available",available_memory," GB ")
            if(current_memory > available_memory*0.5):
                print("You may have to switch off fastMode to conserve GPU memory")
    

# Define operators        
        
    # Projection operator
    def Psub(self,x,indices,slc, mode= 'Normal'):
        out = 0*x
        if(self.params["fastMode"]):
            if mode == 'Normal':
                out = self.toep_ob(x, self.dcomp_kernel[indices].to(self.gpu),smaps=self.smapT[...,slc].to(self.gpu))
            elif mode == 'train':
                out = self.toep_ob(x, self.dcomp_kernel_trn[indices],smaps=self.smapT[...,slc].to(self.gpu))
            elif mode == 'validation':
                out = self.toep_ob(x, self.dcomp_kernel_val[indices],smaps=self.smapT[...,slc].to(self.gpu))
        else:
            for i in range(self.nsl):
                out = self.toep_ob(x, self.dcomp_kernel[indices].to(self.gpu),smaps=self.smapT[...,slc])
        return out

    # Image energy
    #--------------
    def image_energy_sub(self,x,slc, mode= 'Normal'):
        if mode == 'Normal':
            return torch.norm(x*self.mask[...,slc].to(self.gpu),'fro')
        elif mode == 'train':
            return torch.norm(x*self.maskTrn[...,slc].to(self.gpu),'fro')
        elif mode == 'validation':
            return torch.norm(x*self.maskVal[...,slc].to(self.gpu),'fro')
        #return torch.norm(x,'fro')
    
    # @staticmethod
    def giveJsenseCoilsensitivities(self, coilimages,threshold=0.05):
        nsl = coilimages.shape[3]
        sos = np.sqrt(np.sum(np.abs(coilimages)**2,0))
        mask = sos > threshold*np.max(sos)
        mps = np.zeros(coilimages.shape).astype(complex)

        for i in range(nsl):
            maskslc = scipy.ndimage.morphology.binary_closing(mask[...,i],iterations=20)
            maskslc = np.expand_dims(maskslc,0)
            test = sp.fft(coilimages[...,i],axes=(1,2))
            mpslc = mr.app.JsenseRecon(test,mps_ker_width=12,ksp_calib_width=48,max_iter=10,lamda=0.0,show_pbar=False).run()
            mps[:,:,:,i] = mpslc*maskslc.astype(complex)

        return(mps)
    
    
    
    
# # Define the parameters for the right-skewed normal distribution
# mean, std_dev = indices[-1]/2, indices[-1]
# a, b = (indices[0] - mean) / std_dev, (indices[-1] - mean) / std_dev

# # Generate a right-skewed normal distribution
# # np.random.seed(50)
# distribution = truncnorm(a, b, loc=mean, scale=std_dev)

# # Pick indices based on the distribution
# picked_indices = set()
# while len(picked_indices) < num_val_indices:
#     draw = distribution.rvs(1).astype(int)[0]
#     if draw in indices and draw not in picked_indices:
#         picked_indices.add(draw)