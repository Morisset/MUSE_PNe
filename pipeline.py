#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 09:49:26 2021

@author: christophemorisset
"""


import numpy as np
import matplotlib.pyplot as plt
import pyneb as pn
from pyneb import config, log_, atomicData
from pyneb.utils.misc import parseAtom
pn.config.use_multiprocs()
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter
from pathlib import Path
import h5py

#%%
l_dic = {'4641.0': ('N', 2, '4641A', 1),
         '4651.0' : ('O', 2, '4649A', 1),
         '4659.0' : ('Fe', 3, '4659A', 0),
         '4662.0' : ('O', 2, '4662A', 1),
         '4686.0' : ('He', 2, '4686A', 1),
         '4711.0' : ('Ar', 4, '4711A', 0),
         '4713.0' : ('He', 1, '4713A', 1),
         '4741.0' : ('Ar', 4, '4740A', 0),
         '4861.0' : ('H', 1, '4861A', 1),
         '4959.0' : ('O', 3, '4959A', 0),
         '5200.0' : ('N', 1, '5200A', 0),
         '5343.0' : ('C', 2, '5342A', 1),
         '5519.0' : ('Cl', 3, '5518A', 0),
         '5539.0' : ('Cl', 3, '5538A', 0),
         '5677' : ('N', 2, '5676A', 1),
         '5681.0' : ('N', 2, '5679A', 1),
         '5755.0' : ('N', 2, '5755A', 0),
         '5877' : ('He', 1, '5876A', 1),
         '6301.0' : ('O', 1, '6300A', 0),
         '6313' : ('S', 3, '6312A', 0),
         '6365.0' : ('O', 1, '6364A', 0),
         '6463.0' : ('C', 2, '6461A', 0),
         '6549.0' : ('N', 2, '6548A', 0),
         '6564.0' : ('H', 1, '6563A', 1),
         '6585.0' : ('N', 2, '6584A', 0),
         '6679.0' : ('He', 1, '6678A', 1),
         '6719.0' : ('S', 2, '6716A', 0),
         '6733.0' : ('S', 2, '6731A', 0),
         '7002' : ('O', 4, '7004A', 0),
         '7006.0' : ('Ne', 5, '7005A', 0),
         '7067.0' : ('He', 1, '7065A', 1),
         '7137.0' : ('Ar', 3, '7136A', 0),
         '7321.0' : ('O', 2, '7320A', 0),
         '7332.0' : ('O', 2, '7330A', 0),
         '7532.0' : ('Cl', 4, '7531A', 0),
         '7753.0' : ('Ar', 3, '7751A', 0),
         '7772.0' : ('O', 1, '7771A', 1),
         '7774.0' : ('O', 1, '7773A', 1),
         '7777' : ('O', 1, '7775A', 1),
         '8048.0' : ('Cl', 4, '8046A', 0),
         '8730.0' : ('C', 1, '8728A', 0),
         '8736.0' : ('He', 1, '8733A', 1),
         '8753.0' : ('H', 1, '8750A', 1), 
         '9071.0' : ('S', 3, '9069A', 0),
         '9229.0' : ('H', 1, '9229A', 1)
         }
data_dir = Path('/Users/christophemorisset/DATA/MUSE_Jorge/hektor_fits/ngc6778_maps/ngc6778_long_maps')
#%%
class PipeLine(object):
    
    def __init__(self, 
                 data_dir=data_dir, 
                 name = 'ngc6778'):
        """


        Returns
        -------
        None.

        """
        self.log_ = log_
        
        self.data_dir = data_dir
        self.name = name
        self.MC_done = False
        self.N_MC = None
        
        self.load_obs()
        """
        self.add_MC(N_MC)
        self.red_cor_obs()
        """
        
    def load_obs(self):
        
        self.obs = pn.Observation(corrected = False)
        for lam_str in l_dic:
            l = l_dic[lam_str]
            data_file = self.data_dir / Path('{}_MUSE_{}.fits'.format(self.name, lam_str))
            fits_hdu = fits.open(data_file)[0]
            fits_data = fits_hdu.data
            err_data_file = data_dir / Path('{}_MUSE_{}_error.fits'.format(self.name, lam_str))
            err_fits_hdu = fits.open(err_data_file)[0]
            err_fits_data = err_fits_hdu.data
            if l[3] == 1:
                perm = 'r'
            else:
                perm = ''
            label='{}{}{}_{}'.format(l[0], l[1], perm, l[2])
            line = pn.EmissionLine( label=label,
                                   obsIntens=fits_data.ravel(), 
                                   obsError=err_fits_data.ravel(), 
                                   corrected=False, errIsRelative=False)
            self.obs.addLine(line)
        self.wcs = WCS(fits_hdu.header).celestial
        self.shape_data = fits_data.shape
        self.n_obs = self.obs.n_obs
        self.obs.names = ['N{}N'.format(i) for i in range(self.n_obs)]
        self.n_lines = self.obs.n_lines

    def add_MC(self, N_MC=None):
        
        if not self.MC_done:
            if N_MC is not None:
                self.obs.addMonteCarloObs(N_MC)
                self.shape_data = (self.shape_data[0], self.shape_data[1], N_MC+1)
                self.MC_done = True
                self.N_MC = N_MC
                self.n_obs = self.obs.n_obs
        
    def get_image(self, data=None, label=None, type_='median'):
        
        if label is not None:
            if isinstance(label, tuple):
                return self.get_image(label=label[0], type_=type_) / self.get_image(label=label[1], type_=type_)
            d2return = self.obs.getIntens()[label]
        else:
            d2return = data
        if self.N_MC is None:
            return d2return.reshape(self.shape_data)
        else:
            if type_ == 'median':
                return np.nanmedian(d2return.reshape(self.shape_data), 2)
            elif type_ == 'std':
                return np.nanstd(d2return.reshape(self.shape_data), 2)
            elif type_ == 'orig':
                return d2return.reshape(self.shape_data)[:,:,0]
            else:
                self.log_.error('type_ must be one of median, std, orig')
        
    def red_cor_obs(self, plot_=True, label1="H1r_6563A", label2="H1r_4861A", r_theo=2.85):
        
        self.obs.def_EBV(label1=label1, label2=label2, r_theo=r_theo)
        self.obs.correctData()
        if plot_:
            c_Hbeta = self.get_image(self.obs.extinction.cHbeta)
            f, ax = plt.subplots(subplot_kw={'projection': self.wcs}, figsize=(8,8))
            im = ax.imshow(c_Hbeta, vmin=0.5, vmax=.8)
            cb = f.colorbar(im, ax=ax)
            ax.set_title('cHbeta');    
    
    def plot(self, ax=None, data=None, label=None, type_='median', **kwargs):
        
        if ax is None:
            f, ax = plt.subplots(subplot_kw={'projection': self.wcs}, figsize=(8,8))
        else:
            f = plt.gcf()
        im=ax.imshow(self.get_image(data=data, label=label, type_=type_), **kwargs)
        cb = f.colorbar(im, ax=ax)
        ax.set_title('{}'.format(label))  
        
        
    def plot_SN(self, ax=None, data=None, label=None, **kwargs ):

        if ax is None:
            f, ax = plt.subplots(subplot_kw={'projection': self.wcs}, figsize=(8,8))
        else:
            f = plt.gcf()
        med = self.get_image(data=data, label=label, type_='median')
        std = self.get_image(data=data, label=label, type_='std')
        im=ax.imshow(med / std, **kwargs)
        cb = f.colorbar(im, ax=ax)
        ax.set_title('S/N {}'.format(label))  
        
        
        
#%%



