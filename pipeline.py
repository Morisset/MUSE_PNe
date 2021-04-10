#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 09:49:26 2021

@author: christophemorisset
"""


import numpy as np
import matplotlib.pyplot as plt
import pyneb as pn
pn.config.use_multiprocs()
from astropy.io import fits
from astropy.wcs import WCS
from pathlib import Path
try:
    from ai4neb import manage_RM
    AI4NEB_INSTALLED = True
except:
    AI4NEB_INSTALLED = False

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
                 name = 'ngc6778_MUSE_',
                 error_str='_error'):
        """


        Returns
        -------
        None.

        """
        self.log_ = pn.log_
        
        self.data_dir = data_dir
        self.name = name
        self.error_str = error_str
        self.MC_done = False
        self.N_MC = None
        self.TeNe = {}
        
        self.load_obs()
        """
        self.add_MC(N_MC)
        self.red_cor_obs()
        """
        
    def load_obs(self):
        
        self.obs = pn.Observation(corrected = False)
        for lam_str in l_dic:
            l = l_dic[lam_str]
            data_file = self.data_dir / Path('{}{}.fits'.format(self.name, lam_str))
            fits_hdu = fits.open(data_file)[0]
            fits_data = fits_hdu.data
            err_data_file = data_dir / Path('{}{}{}.fits'.format(self.name, lam_str, self.error_str))
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
        
    def get_image(self, data=None, label=None, type_='median', returnObs=False):
        
        if label is not None:
            if isinstance(label, tuple):
                return self.get_image(label=label[0], type_=type_) / self.get_image(label=label[1], type_=type_)
            d2return = self.obs.getIntens(returnObs=returnObs)[label]
        else:
            d2return = data
        if self.N_MC is None:
            return d2return.reshape(self.shape_data)
        else:
            if type_ == 'median':
                return np.nanmedian(d2return.reshape(self.shape_data), 2)
            if type_ == 'mean':
                return np.nanmean(d2return.reshape(self.shape_data), 2)
            elif type_ == 'std':
                return np.nanstd(d2return.reshape(self.shape_data), 2)
            elif type_ == 'orig':
                return d2return.reshape(self.shape_data)[:,:,0]
            else:
                self.log_.error('type_ must be one of median, mean, std, or orig')
        
    def red_cor_obs(self, EBV_min=None, plot_=True, label1="H1r_6563A", label2="H1r_4861A", r_theo=2.85,
                    **kwargs):
        
        self.obs.def_EBV(label1=label1, label2=label2, r_theo=r_theo)
        if EBV_min is not None:
            mask = self.obs.extinction.E_BV < EBV_min
            pn.log_.message('number of spaxels with EBV < {} : {}'.format(EBV_min, mask.sum()),
                            calling='PipeLine.red_cor_obs')
            self.obs.extinction.E_BV[mask] = 0.
        self.obs.correctData()
        if plot_:
            self.plot(data=self.obs.extinction.cHbeta, **kwargs)

    def get_mask_SN(self, label, SN_cut):
        """
        Return a mask True where 1./error[label] < SN_cut
        """
        if isinstance(label, tuple):
            mask = np.zeros_like(self.get_image(data = 1./self.obs.getError()[label[0]], type_='orig'), dtype=bool)
            for i, label1 in enumerate(label):
                if isinstance(SN_cut, tuple):
                    SN_cut1 = SN_cut[i]
                else:
                    SN_cut1 = SN_cut
                mask = mask | self.get_mask_SN(label1, SN_cut1)
        else:
            mask = self.get_image(data = 1./self.obs.getError()[label], type_='orig') < SN_cut
        
        return mask

    def plot(self, ax=None, data=None, label=None, image=None, type_='median', 
             title=None, label_cut=None, SN_cut=None, use_log=False, returnObs=False, 
             interpolation='none', **kwargs):
        
        if ax is None:
            f, ax = plt.subplots(subplot_kw={'projection': self.wcs}, figsize=(8,8))
        else:
            f = plt.gcf()
        if image is None:
            this_image = self.get_image(data=data, label=label, type_=type_, 
                                        returnObs=returnObs)
        else:
            this_image = image
        if SN_cut is not None:
            if label_cut is None:
                label_cut = label
            this_image[self.get_mask_SN(label_cut, SN_cut)] = np.nan
        if use_log:
            with np.errstate(divide='ignore'):
                this_image = np.log10(this_image)
        im=ax.imshow(this_image, interpolation=interpolation, **kwargs)
        cb = f.colorbar(im, ax=ax)
        if title is None:
            if isinstance(label, tuple):
                this_title = '{} / {} ({})'.format(label[0], label[1], type_)
            else:
                this_title = '{} ({})'.format(label, type_)
        else:
            this_title = title
        ax.set_title(this_title)  
        
        
    def plot_SN(self, ax=None, data=None, label=None, title=None, **kwargs ):

        if ax is None:
            f, ax = plt.subplots(subplot_kw={'projection': self.wcs}, figsize=(8,8))
        else:
            f = plt.gcf()
        med = self.get_image(data=data, label=label, type_='median')
        std = self.get_image(data=data, label=label, type_='std')
        with np.errstate(divide='ignore'):
            this_image = med / std
        im=ax.imshow(this_image, **kwargs)
        cb = f.colorbar(im, ax=ax)
        if title is None:
            this_title = 'S/N {}'.format(label)
        else:
            this_title = title
        ax.set_title(this_title)  
        
    def plot_STD(self, ax=None, data=None, label=None, title=None, norm=True, **kwargs ):

        if ax is None:
            f, ax = plt.subplots(subplot_kw={'projection': self.wcs}, figsize=(8,8))
        else:
            f = plt.gcf()
        med = self.get_image(data=data, label=label, type_='median')
        std = self.get_image(data=data, label=label, type_='std')
        with np.errstate(divide='ignore'):
            if norm:
                this_image = std / med * 100
            else:
                this_image = std
        im=ax.imshow(this_image, **kwargs)
        cb = f.colorbar(im, ax=ax)
        if title is None:
            if norm:
                norm_str = '[%]'
            else:
                norm_str = ''
            this_title = 'STD {} {}'.format(label, norm_str)
        else:
            this_title = title
        ax.set_title(this_title)  
        
    def make_diags(self, use_ANN=True):
        
        self.diags = pn.Diagnostics()
        self.diags.ANN_inst_kwargs['verbose']=True
        self.diags.addDiagsFromObs(self.obs)       
        
        
    def add_gCTD(self, label, diag1, diag2, use_ANN=True, limit_res=True, force=False, save=True, **kwargs):
        
        if not AI4NEB_INSTALLED and use_ANN:
            self.log_.error('ai4neb not installed')
        if force:
            ANN = None        
        else:
            ANN = manage_RM(RM_filename=label)        
            if not ANN.model_read:
                ANN = None
        Te, Ne = self.diags.getCrossTemDen(diag1, diag2, obs=self.obs, use_ANN=use_ANN, ANN=ANN,
                                           limit_res=limit_res, **kwargs)
        if use_ANN and ANN is None and save:
            self.diags.ANN.save_RM(filename=label, save_train=True, save_test=True)
        self.TeNe[label] = {'Te': Te, 'Ne': Ne}
    
    def add_T_He(self):
        """
        Zhang 2005
        """
#%%



