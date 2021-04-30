#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 09:49:26 2021

@author: christophemorisset
"""


import numpy as np
import matplotlib.pyplot as plt
import pyneb as pn
from pyneb.utils.misc import parseAtom, int_to_roman
pn.config.use_multiprocs()
from pathlib import Path
from scipy.interpolate import interp1d
import pandas as pd
try:
    from ai4neb import manage_RM
    AI4NEB_INSTALLED = True
except:
    AI4NEB_INSTALLED = False
import os
import pickle
import gzip
import pymysql

#%% Define dictionnaries hektor to pyneb
l_dics = {'NGC6778': {'4641.0' : ('N', 3, '4641A', 1),
                      '4651.0' : ('O', 2, '4649.13A', 1),
                     '4662.0' : ('O', 2, '4661.63A', 1),
                     '4686.0' : ('He', 2, '4686A', 1),
                     '4711.0' : ('Ar', 4, '4711A', 0),
                     '4713.0' : ('He', 1, '4713A', 1),
                     '4741.0' : ('Ar', 4, '4740A', 0),
                     '4861.0' : ('H', 1, '4861A', 1),
                     '4959.0' : ('O', 3, '4959A', 0),
                     '5200.0' : ('N', 1, '5199A+', 0),
                     '5343.0' : ('C', 2, '5342.0A', 1),
                     '5519.0' : ('Cl', 3, '5518A', 0),
                     '5539.0' : ('Cl', 3, '5538A', 0),
                     '5677' : ('N', 2, '5676.02A', 1),
                     '5681.0' : ('N', 2, '5679.56A', 1),
                     '5755.0' : ('N', 2, '5755A', 0),
                     '5877' : ('He', 1, '5876A', 1),
                     '6301.0' : ('O', 1, '6300A', 0),
                     '6313' : ('S', 3, '6312A', 0),
                     '6365.0' : ('O', 1, '6364A', 0),
                     '6463.0' : ('C', 2, '6462.0A', 1),
                     '6549.0' : ('N', 2, '6548A', 0),
                     '6564.0' : ('H', 1, '6563A', 1),
                     '6585.0' : ('N', 2, '6584A', 0),
                     '6679.0' : ('He', 1, '6678A', 1),
                     '6719.0' : ('S', 2, '6716A', 0),
                     '6733.0' : ('S', 2, '6731A', 0),
                     '7006.0' : ('Ar', 5, '7005A', 0),
                     '7067.0' : ('He', 1, '7065A', 1),
                     '7137.0' : ('Ar', 3, '7136A', 0),
                     '7283' : ('He', 1, '7281A', 1),
                     '7321.0' : ('O', 2, '7319A+', 0),
                     '7332.0' : ('O', 2, '7330A+', 0),
                     '7532.0' : ('Cl', 4, '7531A', 0),
                     '7753.0' : ('Ar', 3, '7751A', 0),
                     '7772.0' : ('O', 1, '7771A', 1),
                     '7774.0' : ('O', 1, '7773A', 1),
                     '7777' : ('O', 1, '7775A', 1),
                     '8048.0' : ('Cl', 4, '8046A', 0),
                     '8730.0' : ('C', 1, '8728A', 0),
                     '8753.0' : ('H', 1, '8750A', 1), 
                     '9071.0' : ('S', 3, '9069A', 0),
                     '9229.0' : ('H', 1, '9229A', 1)
                     },
           'HF22': {'4641.0' : ('N', 3, '4641A', 1),
                      '4651.0' : ('O', 2, '4649.13A', 1),
                     '4662.0' : ('O', 2, '4661.63A', 1),
                     '4686.0' : ('He', 2, '4686A', 1),
                     '4711.0' : ('Ar', 4, '4711A', 0),
                     '4713.0' : ('He', 1, '4713A', 1),
                     '4741.0' : ('Ar', 4, '4740A', 0),
                     '4861.0' : ('H', 1, '4861A', 1),
                     '4959.0' : ('O', 3, '4959A', 0),
                     '5200.0' : ('N', 1, '5199A+', 0),
                     '5343.0' : ('C', 2, '5342.0A', 1),
                     '5519.0' : ('Cl', 3, '5518A', 0),
                     '5539.0' : ('Cl', 3, '5538A', 0),
                     '5677' : ('N', 2, '5676.02A', 1),
                     '5681.0' : ('N', 2, '5679.56A', 1),
                     '5755.0' : ('N', 2, '5755A', 0),
                     '5877' : ('He', 1, '5876A', 1),
                     '6301.0' : ('O', 1, '6300A', 0),
                     '6313' : ('S', 3, '6312A', 0),
                     '6365.0' : ('O', 1, '6364A', 0),
                     '6463.0' : ('C', 2, '6462.0A', 1),
                     '6549.0' : ('N', 2, '6548A', 0),
                     '6564.0' : ('H', 1, '6563A', 1),
                     '6585.0' : ('N', 2, '6584A', 0),
                     '6679.0' : ('He', 1, '6678A', 1),
                     '6719.0' : ('S', 2, '6716A', 0),
                     '6733.0' : ('S', 2, '6731A', 0),
                     '7006.0' : ('Ar', 5, '7005A', 0),
                     '7067.0' : ('He', 1, '7065A', 1),
                     '7137.0' : ('Ar', 3, '7136A', 0),
                     '7283' : ('He', 1, '7281A', 1),
                     '7321.0' : ('O', 2, '7319A+', 0),
                     '7332.0' : ('O', 2, '7330A+', 0),
                     '7532.0' : ('Cl', 4, '7531A', 0),
                     '7753.0' : ('Ar', 3, '7751A', 0),
                     '7772.0' : ('O', 1, '7771A', 1),
                     '7774.0' : ('O', 1, '7773A', 1),
                     '7777' : ('O', 1, '7775A', 1),
                     '8048.0' : ('Cl', 4, '8046A', 0),
                     '8730.0' : ('C', 1, '8728A', 0),
                     '8753.0' : ('H', 1, '8750A', 1), 
                     '9071.0' : ('S', 3, '9069A', 0),
                     '9229.0' : ('H', 1, '9229A', 1)
                     },          
           'M142': {'4640.0' : ('N', 3, '4641A', 1),
                 '4648.0' : ('O', 2, '4649.13A', 1),
                 '4660.0' : ('O', 2, '4661.63A', 1),
                 '4684.0' : ('He', 2, '4686A', 1),
                 '4710.0' : ('Ar', 4, '4711A', 0),
                 '4711.0' : ('He', 1, '4713A', 1),
                 '4739.0' : ('Ar', 4, '4740A', 0),
                 '4860.0' : ('H', 1, '4861A', 1),
                 '4958.0' : ('O', 3, '4959A', 0),
                 '5198.0' : ('N', 1, '5199A+', 0),
                 '5343.0' : ('C', 2, '5342.0A', 1),
                 '5516.0' : ('Cl', 3, '5518A', 0),
                 '5536.0' : ('Cl', 3, '5538A', 0),
                 '5677' : ('N', 2, '5679.56A', 1),
                 '5753.0' : ('N', 2, '5755A', 0),
                 '5874' : ('He', 1, '5876A', 1),
                 '6299.0' : ('O', 1, '6300A', 0),
                 '6310' : ('S', 3, '6312A', 0),
                 '6362.0' : ('O', 1, '6364A', 0),
                 '6460.0' : ('C', 2, '6462.0A', 1),
                 '6546.0' : ('N', 2, '6548A', 0),
                 '6561.0' : ('H', 1, '6563A', 1),
                 '6582.0' : ('N', 2, '6584A', 0),
                 '6677.0' : ('He', 1, '6678A', 1),
                 '6715.0' : ('S', 2, '6716A', 0),
                 '6729.0' : ('S', 2, '6731A', 0),
                 '7006.0' : ('Ar', 5, '7005A', 0),
                 '7063.0' : ('He', 1, '7065A', 1),
                 '7134.0' : ('Ar', 3, '7136A', 0),
                 '7280.0' : ('He', 1, '7281A', 1),
                 '7318.0' : ('O', 2, '7319A+', 0),
                 '7328.0' : ('O', 2, '7330A+', 0),
                 '7528.0' : ('Cl', 4, '7531A', 0),
                 '7749.0' : ('Ar', 3, '7751A', 0),
                 '7770.0' : ('O', 1, '7771A', 1),
                 '7772.0' : ('O', 1, '7773A', 1),
                 '7773' : ('O', 1, '7775A', 1),
                 '8044.0' : ('Cl', 4, '8046A', 0),
                 '8725.0' : ('C', 1, '8728A', 0),
                 '8748.0' : ('H', 1, '8750A', 1), 
                 '9067.0' : ('S', 3, '9069A', 0),
                 '9227.0' : ('H', 1, '9229A', 1)
                 }
          }

#%% Rename files from Hektor to PyNeb

def rename_files(obj_name):
    
    """
    example:
        rename_files(obj_name = 'NGC6778')
    """
    data_dir = Path(os.environ['MUSE_DATA']) / Path('{}/maps'.format(obj_name))
    name = '{}_MUSE_'.format(obj_name)
    newname = '{}_MUSE_b_'.format(obj_name)
    
    for lam_str in l_dics[obj_name]:
        l = l_dics[obj_name][lam_str]
        if l[3] == 1:
            rec_str = 'r'
        else:
            rec_str = ''
        data_file = data_dir / Path('{}{}.fits'.format(name, lam_str))
        new_data_file = data_dir / Path('{}{}{}{}_{}.fits'.format(newname, l[0], l[1], rec_str, l[2]))
        edata_file1 = data_dir / Path('{}{}_error.fits'.format(name, lam_str))
        new_edata_file1 = data_dir / Path('{}{}{}{}_{}_error.fits'.format(newname, l[0], l[1], rec_str, l[2]))
        edata_file2 = data_dir / Path('{}{}_error_alfalike.fits'.format(name, lam_str))
        new_edata_file2 = data_dir / Path('{}{}{}{}_{}_error_alfalike.fits'.format(newname, l[0], l[1], rec_str, l[2]))
        if data_file.exists():
            print(data_file, '->', new_data_file)
            data_file.rename(new_data_file)
        if edata_file1.exists():
            print(edata_file1, '->', new_edata_file1)
            edata_file1.rename(new_edata_file1)
        if edata_file2.exists():
            print(edata_file2, '->', new_edata_file2)
            edata_file2.rename(new_edata_file2)

#%% get_label_str

def get_label_str(label):
    
    
    lab1 = label.split('_')[0]
    lab3 = label.split('_')[1]
    if lab1[-1] == 'r':
        lab1 = '{}'.format(lab1[0:-1])
        forb = False
    else:
        forb = True
        
    lab1, lab2 = parseAtom(lab1)

    lab2 = int_to_roman(int(lab2))
    
    if forb:
        lab1 = '[{} {}]'.format(lab1, lab2)
    else:
        lab1 = '{} {}'.format(lab1, lab2)
        
    if lab3[-1] == '+':
        lab3 = lab3[0:-1]
        blend_str = '+'
    else:
        blend_str=''
    if lab3[-1] == 'A':
        lab3 = lab3[0:-1]
    lab3 = lab3.split('.')[0]
    return '{} {}{}'.format(lab1, lab3, blend_str)

#%% Get from 3MdB

def get_ICFs_3MdB():
    
    host = os.environ['MdB_HOST']
    user = os.environ['MdB_USER']
    passwd = os.environ['MdB_PASSWD']
    port = os.environ['MdB_PORT']
    
    
    db = pymysql.connect(host=host, user=user, passwd=passwd, port=int(port), db='3MdB_17')
    res = pd.read_sql("""SELECT 
A_HYDROGEN_vol_1 as Hp,
A_HELIUM_vol_1 as Hep,
A_HELIUM_vol_2 as Hepp,
A_CARBON_vol_1 as Cp,
A_NITROGEN_vol_1 as Np,
A_OXYGEN_vol_1 as Op,
A_OXYGEN_vol_2 as Opp,
A_SULPHUR_vol_1 as Sp,
A_SULPHUR_vol_2 as Spp,
A_CHLORINE_vol_2 as Clpp,
A_CHLORINE_vol_3 as Clppp,
A_ARGON_vol_2 as Arpp,
A_ARGON_vol_3 as Arppp
FROM abion_17
WHERE abion_17.ref = 'PNe_2020'
""", con=db)
    db.close()
    res['Hep_pp'] = np.log10(res.Hep / res.Hepp)
    res['Op_pp'] = np.log10(res.Op / res.Opp)
    res['Sp_pp'] = np.log10(res.Sp / res.Spp)
    res['Clpp_ppp'] = np.log10(res.Clpp / res.Clppp)
    res['Arpp_ppp'] = np.log10(res.Arpp / res.Arppp)
    
    res.to_csv('PN2020_3MdB_MUSE.csv.gz')
    
#%% Pipeline
class PipeLine(object):
    
    def __init__(self, 
                 data_dir, 
                 obj_name,
                 error_str='_error', 
                 err_default=0.0,
                 flux_normalisation=1.0,
                 cmap='viridis', 
                 random_seed=None):
        """


        Returns
        -------
        None.

        """
        self.log_ = pn.log_
        
        self.data_dir = data_dir
        self.obj_name = obj_name
        self.error_str = error_str
        self.err_default = err_default
        self.MC_done = False
        self.N_MC = None
        self.TeNe = {}
        self.NII_corrected = False
        self.OII_corrected = False
        self.atom_dic  = {}
        self.abund_dic =  {}
        self.cmap = cmap
        np.random.seed(random_seed)
        self.ANN_inst_kwargs = {'RM_type' : 'SK_ANN', 
                                'verbose' : False, 
                                'scaling' : True,
                                'use_log' : False,
                                'random_seed' : None
                                }
        self.ANN_init_kwargs = {'solver' : 'lbfgs', 
                                'activation' : 'tanh', 
                                'hidden_layer_sizes' : (10, 10), 
                                'max_iter' : 20000
                                }
        self.flux_normalisation = flux_normalisation
        self.load_obs()
        """
        self.add_MC(N_MC)
        self.red_cor_obs()
        """
        
        
    def load_obs(self, clean_error=1e-5, mask=None):
        
        obs_name = Path(self.data_dir) / Path('{}_MUSE_b_*.fits'.format(self.obj_name))
        self.obs = pn.Observation(obs_name, fileFormat='fits_IFU', 
                                  corrected = False, 
                                  errStr=self.error_str, 
                                  errIsRelative=False,
                                  err_default=self.err_default,
                                  addErrDefault = True)
        
        
        int_file = Path(self.data_dir) / Path('{}_int_line_fluxes.dat'.format(self.obj_name))
        self.obs_int = pn.Observation(int_file, fileFormat='lines_in_rows_err_cols',
                                      corrected = False, 
                                      errIsRelative=False,
                                      err_default=self.err_default,
                                      addErrDefault = True)
        
        for line in self.obs_int.getSortedLines():
            line.obsIntens *= self.flux_normalisation / self.obs.n_obs
        obs_int_dic = self.obs_int.getIntens(returnObs=True)
        err_int_dic = self.obs_int.getError(returnObs=True)
        for line in self.obs.getSortedLines():
            line.obsIntens *= self.flux_normalisation
            mask_err = np.abs(line.obsError - self.err_default) < clean_error
            line.obsIntens[mask_err] = np.nan
            try:
                line.obsIntens[0] = obs_int_dic[line.label][0]
                line.obsError[0] = err_int_dic[line.label][0]
            except:
                self.log_.warn('Integrated value for {} not done'.format(line.label),
                                calling='PipeLine.load_obs')
            
        
        
        self.n_obs = self.obs.n_obs
        
    def add_MC(self, N_MC=None):
        
        if not self.MC_done:
            if N_MC is not None:
                self.obs.addMonteCarloObs(N_MC)
                self.MC_done = True
                self.N_MC = self.obs.N_MC
                self.n_obs = self.obs.n_obs
        
    def get_image(self, data=None, label=None, type_='median', returnObs=False):
        
        if label is not None:
            if isinstance(label, tuple):
                with np.errstate(divide='ignore', invalid='ignore'):
                    to_return = self.get_image(label=label[0], type_=type_) / self.get_image(label=label[1], type_=type_)
                return to_return
            d2return = self.obs.getIntens(returnObs=returnObs)[label]
        else:
            d2return = data
        if self.N_MC is None:
            return d2return.reshape(self.obs.data_shape)
        else:
            if type_ == 'median':
                return np.nanmedian(d2return.reshape(self.obs.data_shape), 2)
            if type_ == 'mean':
                return np.nanmean(d2return.reshape(self.obs.data_shape), 2)
            elif type_ == 'std':
                return np.nanstd(d2return.reshape(self.obs.data_shape), 2)
            elif type_ == 'orig':
                return d2return.reshape(self.obs.data_shape)[:,:,0]
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
             interpolation='none', cb_title=None,  mask=None, **kwargs):
        
        if ax is None:
            f, ax = plt.subplots(subplot_kw={'projection': self.obs.wcs}, figsize=(8,8))
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
        if mask is not None:
            this_image[mask] = np.nan
        if use_log:
            with np.errstate(divide='ignore', invalid='ignore'):
                this_image = np.log10(this_image)
        im=ax.imshow(this_image, interpolation=interpolation, cmap=self.cmap, **kwargs)

        cb = f.colorbar(im, ax=ax, fraction=0.048, pad=0.0)
        cb.ax.get_yaxis().labelpad = 20
        cb.ax.set_ylabel(cb_title, rotation=270)
        if title is None:
            if isinstance(label, tuple):
                this_title = '{} / {} ({})'.format(label[0], label[1], type_)
            else:
                this_title = '{} ({})'.format(label, type_)
        else:
            this_title = title
        ax.set_title(this_title)
        ax.set_xlabel('Right Ascension')
        ax.set_ylabel('Declination')

        
        
    def plot_SN(self, ax=None, data=None, label=None, title=None, **kwargs ):

        if ax is None:
            f, ax = plt.subplots(subplot_kw={'projection': self.obs.wcs}, figsize=(8,8))
        else:
            f = plt.gcf()
        med = self.get_image(data=data, label=label, type_='median')
        std = self.get_image(data=data, label=label, type_='std')
        with np.errstate(divide='ignore'):
            this_image = med / std
        im=ax.imshow(this_image, cmap=self.cmap, **kwargs)
        cb = f.colorbar(im, ax=ax)
        if title is None:
            this_title = 'Med/STD {}'.format(label)
        else:
            this_title = title
        ax.set_title(this_title)  
        
    def plot_STD(self, ax=None, data=None, label=None, title=None, norm=True, **kwargs ):

        if ax is None:
            f, ax = plt.subplots(subplot_kw={'projection': self.obs.wcs}, figsize=(8,8))
        else:
            f = plt.gcf()
        med = self.get_image(data=data, label=label, type_='median')
        std = self.get_image(data=data, label=label, type_='std')
        with np.errstate(divide='ignore'):
            if norm:
                this_image = std / med * 100
            else:
                this_image = std
        im=ax.imshow(this_image, cmap=self.cmap, **kwargs)
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
                                           limit_res=limit_res, end_tem=30000, **kwargs)
        if use_ANN and ANN is None and save:
            self.diags.ANN.save_RM(filename=label, save_train=True, save_test=True)
        self.TeNe[label] = {'Te': Te, 'Ne': Ne}
    
    def add_T_He(self):
        """
        Mendez Delgado 2021
        """
        self.TeNe['He1'] = {}
        dens = np.asarray((100,   500,  1000,  2000,  3000,  4000,  5000,  6000,  7000,
                           8000,  9000, 10000, 12000, 15000, 20000, 25000, 30000, 40000,
                           45000, 50000))
        alpha = np.asarray((92984, 81830, 77896, 69126, 65040, 62517, 60744, 59402, 58334,
                            57456, 56715, 56077, 55637, 55087, 54364, 53796, 53329, 52591,
                            52289, 52019))
        beta = np.asarray((-7455, -6031, -5527, -4378, -3851, -3529, -3305 ,-3137, -3004, 
                           -2895, -2804, -2726, -2676, -2611, -2523, -2452, -2392, -2297, 
                           -2257, -2222))
        alpha_int = interp1d(dens, alpha, bounds_error=False)
        beta_int = interp1d(dens, beta, bounds_error=False)
        
        alphas = alpha_int(self.TeNe['N2S2']['Ne'])
        betas = beta_int(self.TeNe['N2S2']['Ne'])
        with np.errstate(divide='ignore', invalid='ignore'):
            R_He = self.obs.getIntens()['He1r_7281A'] / self.obs.getIntens()['He1r_6678A']
        
        Te = alphas * R_He + betas
        Te[np.isinf(Te)] = np.nan
        
        self.TeNe['He1']['Te'] = Te
        self.log_.message('Done', calling='Pipeline.add_T_He')
            

    def add_T_PJ(self, den=1e3, Hep=0.095, Hepp = 0.005):
    
        cont = pn.Continuum()
        tab_tem = np.linspace(1000, 30000, 100)
        tab_den = np.ones_like(tab_tem) * den
        tab_Hep = np.ones_like(tab_tem) * Hep
        tab_Hepp = np.ones_like(tab_tem) * Hepp
    
        tab_PJ =  cont.BJ_HI(tab_tem, tab_den, tab_Hep, tab_Hepp, wl_bbj = 8100, wl_abj = 8400, HI_label='9_3')
        tem_inter = interp1d(tab_PJ, tab_tem, bounds_error=False)
    
        self.TeNe['PJ'] = {}
        C_8100 = self.obs.getIntens()['H1r_8100.0']
        C_8400 = self.obs.getIntens()['H1r_8400.0']
        HI = self.obs.getIntens()['H1r_9229A']
        with np.errstate(divide='ignore', invalid='ignore'):
            PJ_HI = (C_8100 - C_8400) /  HI
        self.TeNe['PJ']['Te'] = tem_inter(PJ_HI)
        self.log_.message('Done', calling='Pipeline.add_T_PJ')
               
    def _make_grid_TPJ(self, tem_min=2000, tem_max=30000, 
                       log_den_min=2, log_den_max=4, 
                       Hep_min = 0.0, Hep_max = 1.0, 
                       HeoH=0.1):
        
        N = 5000
        tab_tem = tem_min + np.random.rand(N) * (tem_max - tem_min)
        tab_log_den = log_den_min + np.random.rand(N) * (log_den_max - log_den_min)
        tab_Hep = HeoH * (Hep_min + np.random.rand(N) * (Hep_max - Hep_min))
        tab_Hepp = HeoH - tab_Hep
        
        cont = pn.Continuum()
        tab_PJ =  cont.BJ_HI(tab_tem, 10**tab_log_den, tab_Hep, tab_Hepp, wl_bbj = 8100, wl_abj = 8400, HI_label='9_3')
        df = pd.DataFrame({'PJ':tab_PJ, 'tem':tab_tem, 'log_den':tab_log_den, 'Hep':tab_Hep})
        df.to_csv('PJ_data.csv')

    def _train_ML_PJ(self):
        
        try:
            df = pd.read_csv('PJ_data.csv')
        except:
            self._make_grid_TPJ()
            df = pd.read_csv('PJ_data.csv')
        X = df[['PJ', 'log_den', 'Hep']]
        y = np.log10(df[['tem']])
        self.ANN = manage_RM(X_train=X, y_train=y, **self.ANN_inst_kwargs)
        self.ANN.init_RM(**self.ANN_init_kwargs)
        self.ANN.train_RM()

    def add_T_PJ_ML(self):
    
        self.TeNe['PJ_ANN'] = {}
        C_8100 = self.obs.getIntens()['H1r_8100.0']
        C_8400 = self.obs.getIntens()['H1r_8400.0']
        HI = self.obs.getIntens()['H1r_9229A']
    
        with np.errstate(divide='ignore', invalid='ignore'):
            PJ_HI = (C_8100 - C_8400) /  HI
            log_den = np.log10(self.TeNe['N2S2']['Ne'])
        Hep = 0.1 * (self.abund_dic['He1r_6678A'] / (self.abund_dic['He1r_6678A'] + self.abund_dic['He2r_4686A']))
        
        self._train_ML_PJ()
        mask = np.isfinite(PJ_HI) & np.isfinite(log_den) & np.isfinite(Hep)
        self.ANN.set_test(np.array((PJ_HI[mask], log_den[mask], Hep[mask])).T)
        
        self.ANN.predict()
        pred = np.zeros_like(log_den) * np.nan
        pred[mask] = self.ANN.pred
        self.TeNe['PJ_ANN']['Te'] = 10**pred
        self.log_.message('Done', calling='Pipeline.add_T_PJ_ML')
        
            
    def set_abunds(self, IP_cut = 17, label=None, tem_HI=None):
        
        Hbeta = self.obs.getIntens()['H1r_4861A']
        
        
        for line in self.obs.getSortedLines():
            if label is None or line.label == label: 
                if line.atom not in self.atom_dic:
                    if line.atom[-1] == 'r':
                        atom = pn.RecAtom(line.elem, line.spec, case='A')
                        IP = pn.utils.physics.IP[atom.elem][atom.spec-1]
                    else:
                        atom = pn.Atom(line.elem, line.spec)
                        if atom.spec-2 < 0:
                            IP = 0.
                        else:
                            IP = pn.utils.physics.IP[atom.elem][atom.spec-2]
                    self.atom_dic[line.atom] = (atom, IP)
                else:
                    atom, IP = self.atom_dic[line.atom]
                
                if IP < IP_cut:
                    Te = self.TeNe['N2S2']['Te']
                    Ne = self.TeNe['N2S2']['Ne']
                else:
                    Te = self.TeNe['S3S2']['Te']
                    Ne = self.TeNe['S3S2']['Ne']
                if line.is_valid:
                    self.abund_dic[line.label] = atom.getIonAbundance(line.corrIntens/Hbeta, Te, Ne, 
                                                                      to_eval=line.to_eval, Hbeta=1.,
                                                                      tem_HI=tem_HI)
                else:
                    self.abund_dic[line.label] = None
                self.log_.message('Abund from {} done.'.format(line.label), calling='PipeLine.set_abunds')

        
    def correc_NII(self, tem, den=1e3):
        
        if self.NII_corrected:
            print('Already corrected')
            return
        if tem is None:
            return
        I_5755 = self.obs.getIntens()['N2_5755A']
        I_5679 = self.obs.getIntens()['N2r_5679.56A']        
        pn.atomicData.setDataFile('n_ii_rec_P91.func')
        N2rP = pn.RecAtom('N', 2, case='B')
        pn.atomicData.setDataFile('n_ii_rec_FSL11.func')
        N2rF = pn.RecAtom('N', 2, case='B')
        R_5755_5679 = N2rP.getEmissivity(tem, den, label='5755.', product=False) / N2rF.getEmissivity(tem, den, label='5679.56', product=False)
        with np.errstate(divide='ignore', invalid='ignore'):
            I_5755_new = I_5755 - R_5755_5679*I_5679
        for line in self.obs.lines:
            if line.label == 'N2_5755A':
                line.corrIntens = I_5755_new
        self.NII_corrected = True


    def correc_OII(self, tem, den=1e3, rec_label='O2r_4661.63A'): 
        """
        rec_label may be 'O2r_4661.63A' or 'O2r_4649.13A'
        
        """
        
        if self.OII_corrected:
            print('Already corrected')
            return
        if tem is None:
            return
        
        I_7320 = self.obs.getIntens()['O2_7319A+']
        I_7330 = self.obs.getIntens()['O2_7330A+']
        I_7325 = I_7320 + I_7330
        
        I_REC = self.obs.getIntens()[rec_label]
        
        pn.atomicData.setDataFile('o_ii_rec_P91.func')
        O2rP = pn.RecAtom('O', 2, case='B') 
        pn.atomicData.setDataFile('o_ii_rec_SSB17-B-opt.hdf5')
        O2rS = pn.RecAtom('O', 2, case='B')
        wave_str = rec_label.split('_')[1][:-1]
        emisP = O2rP.getEmissivity(tem, den, label='7325+', product=False)
        if rec_label == 'O2r_4649.13A':
            emisR = O2rS.getEmissivity(tem, den, label='4649.13', product=False) + O2rS.getEmissivity(tem, den, label='4650.84', product=False)
        else:
            emisR = O2rS.getEmissivity(tem, den, label=wave_str, product=False)
        with np.errstate(divide='ignore', invalid='ignore'):
            R_7325_REC = emisP / emisR
            I_7325_new = I_7325 - R_7325_REC * I_REC

        for line in self.obs.lines:
            with np.errstate(divide='ignore', invalid='ignore'):
                if line.label == 'O2_7319A+':
                    line.corrIntens = I_7325_new * I_7320 / I_7325
                if line.label == 'O2_7330A+':
                    line.corrIntens = I_7325_new * I_7330 / I_7325
        self.OII_corrected = True

    def save_TeNe(self, filename):
        
        with gzip.open(filename, 'wb') as handle:
            pickle.dump(self.TeNe, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.log_.message('Done', calling='Pipeline.save_TeNe')

            
    def read_TeNe(self, filename):
        
        with gzip.open(filename, 'rb') as handle:
            self.TeNe = pickle.load(handle)
        self.log_.message('Done', calling='Pipeline.read_TeNe')
        
    def save_abunds(self, filename):
        
        with gzip.open(filename, 'wb') as handle:
            pickle.dump(self.abund_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.log_.message('Done', calling='Pipeline.save_abunds')
            
    def read_abunds(self, filename):
        
        with gzip.open(filename, 'rb') as handle:
            self.abund_dic = pickle.load(handle)
        self.log_.message('Done', calling='Pipeline.read_abunds')


    def predict_ICF_ML(self, N_train=5, tol=0.5, learning_rate=0.600, max_depth=14, n_estimators=100):
        
        try:
            df = pd.read_csv('PN2020_3MdB_MUSE.csv.gz')
        except:
            get_ICFs_3MdB()
            df = pd.read_csv('PN2020_3MdB_MUSE.csv.gz')
            
        get_ab = lambda label: self.abund_dic[label][0]
                
        Hepp_p = get_ab('He2r_4686A')/get_ab('He1r_6678A')
        Opp_p = get_ab('O3_4959A')/get_ab('O2_7330A+')
        Spp_p = get_ab('S3_9069A')/get_ab('S2_6731A')
        Clppp_pp = get_ab('Cl4_8046A')/get_ab('Cl3_5538A')
        Arppp_pp = get_ab('Ar4_4740A')/get_ab('Ar3_7751A')
        
        mask1 = np.abs( ( np.log10(df['Hepp'] / df['Hep']) - np.log10(Hepp_p) ) ) < tol
        mask2 = np.abs( ( np.log10(df['Opp'] / df['Op']) - np.log10(Opp_p) ) ) < tol
        mask3 = np.abs( ( np.log10(df['Spp'] / df['Sp']) - np.log10(Spp_p) ) ) < tol
        mask4 = np.abs( ( np.log10(df['Clppp'] / df['Clpp']) - np.log10(Clppp_pp) ) ) < tol
        mask5 = np.abs( ( np.log10(df['Arppp'] / df['Arpp']) - np.log10(Arppp_pp) ) ) < tol

                
        mask = mask1 & mask2 & mask3 & mask4 
        if N_train == 5:
            mask = mask & mask5
        print('mask 1: {}, mask 2:{}, mask 3:{}, mask 4:{},, mask 5:{}, mask:{}'.format(mask1.sum(), 
                                                                                        mask2.sum(), 
                                                                                        mask3.sum(),
                                                                                        mask4.sum(),
                                                                                        mask5.sum(),
                                                                                        mask.sum()))
        if N_train == 4:
            X_train=np.array((df['Hepp'][mask]/df['Hep'][mask],
                              df['Opp'][mask]/df['Op'][mask], 
                              df['Spp'][mask]/df['Sp'][mask], 
                              df['Clppp'][mask]/df['Clpp'][mask])).T     
        
        elif N_train == 5:
            X_train=np.array((df['Hepp'][mask]/df['Hep'][mask],
                              df['Opp'][mask]/df['Op'][mask], 
                              df['Spp'][mask]/df['Sp'][mask], 
                              df['Clppp'][mask]/df['Clpp'][mask],
                              df['Arppp'][mask]/df['Arpp'][mask])).T     
               
        icf_c = (1./(df['Cp'][mask]))
        icf_n = (1./(df['Np'][mask]))
        icf_o = (1./(df['Op'][mask]+df['Opp'][mask]))
        icf_s = (1./(df['Sp'][mask]+df['Spp'][mask]))
        icf_cl = (1./(df['Clpp'][mask]+df['Clppp'][mask]))
        icf_ar = (1./(df['Arpp'][mask]+df['Arppp'][mask]))
        
        y_train = np.log10(np.array((icf_c, icf_n, icf_o, icf_s,icf_cl, icf_ar)).T)
        
        RM = manage_RM(RM_type='XGB',
                              X_train=X_train, 
                              y_train=y_train, 
                              scaling=True,
                              scaling_y=False,
                              use_log=True,
                              verbose=True, 
                              random_seed=42,
                              split_ratio=0.2, 
                              clear_session=True,
                              pca_N=0)
        RM.init_RM(learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators)
        RM.train_RM()
        
        if N_train == 4:
            RM.set_test(X = np.expand_dims(np.array((Hepp_p, Opp_p, Spp_p, Clppp_pp)), 0))            
        elif N_train == 5:
            RM.set_test(X = np.expand_dims(np.array((Hepp_p, Opp_p, Spp_p, Clppp_pp, Arppp_pp)), 0))
        RM.predict()
        self.RM = RM
        self.ICF_ML = {'C+': 10**RM.pred[0][0],
                       'N+': 10**RM.pred[0][1],
                       'O+ + O++': 10**RM.pred[0][2],
                       'S+ + S++': 10**RM.pred[0][3],
                       'Cl2+ + Cl3+': 10**RM.pred[0][4],
                       'Ar2+ + Ar3+': 10**RM.pred[0][5]}
                       
    
#%% run pipeline and all

def run_pipeline(obj_name, Te_corr, random_seed=None):
        
    data_dir = Path(os.environ['MUSE_DATA']) / Path('{}/maps'.format(obj_name))

    PL = PipeLine(data_dir = data_dir,
                  obj_name = obj_name, 
                  error_str='error', err_default=0.05,
                  random_seed=random_seed)

    PL.log_.level=3
    
    PL.obs.addSum(('O1r_7771A', 'O1r_7773A', 'O1r_7775A'), 'O1r_7773+')
    
    PL.obs.removeLine('O1r_7771A')
    PL.obs.removeLine('O1r_7773A')
    PL.obs.removeLine('O1r_7775A')
    
    PL.obs.getLine(label='O2r_4649.13A').to_eval = 'L(4649.13) + L(4650.84)'
    
    PL.add_MC(150)
    print('Data shape:', PL.obs.data_shape)
    print('Number of lines , valid ones: ', PL.obs.n_lines,PL.obs.n_valid_lines)
    
    PL.obs.def_EBV()
    PL.red_cor_obs(EBV_min = 0., plot_=False)#, label1="H1r_6561.0", label2="H1r_4860.0")    

    PL.correc_NII(Te_corr)
    PL.correc_OII(Te_corr, rec_label='O2r_4649.13A')    
    
    PL.make_diags()    
    
    PL.diags.ANN_inst_kwargs['verbose']=True
    PL.diags.ANN_init_kwargs['activation']='tanh' #identity’, ‘logistic’, ‘tanh’, ‘relu’
    PL.diags.ANN_init_kwargs['solver']= 'lbfgs' #‘lbfgs’, ‘sgd’, ‘adam’
    PL.diags.ANN_init_kwargs['hidden_layer_sizes'] = (10, 20, 10)
    
    print(PL.n_obs)
    pn.log_.timer('Starting', quiet=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        PL.add_gCTD('N2S2', '[NII] 5755/6548', '[SII] 6731/6716')
        PL.add_gCTD('N2S2_84', '[NII] 5755/6584', '[SII] 6731/6716')
        PL.add_gCTD('S3Cl3', '[SIII] 6312/9069', '[ClIII] 5538/5518')
        PL.add_gCTD('S3S2', '[SIII] 6312/9069', '[SII] 6731/6716')
        PL.add_gCTD('S3Ar4', '[SIII] 6312/9069', '[ArIV] 4740/4711')
    pn.log_.timer('ANN getCrossTemDen done')    
    
    PL.add_T_He()
    
    PL.set_abunds()    
    
    PL.add_T_PJ()
    PL.add_T_PJ_ML()
    
    PL.save_TeNe('{}/PipelineResults/{}_{}_TeNe.pickle.gz'.format(os.environ['MUSE_DATA'], obj_name, Te_corr))
    PL.save_abunds('{}/PipelineResults/{}_{}_abunds.pickle.gz'.format(os.environ['MUSE_DATA'], obj_name, Te_corr))
    
        
def run_all():

    for obj_name in ('M142', ): #'HF22','NGC6778','M142', 
        for Te_corr in (None, 3000, 8000):
            run_pipeline(obj_name, Te_corr, random_seed=42)
