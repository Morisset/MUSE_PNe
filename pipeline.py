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
from scipy.interpolate import interp1d
try:
    from ai4neb import manage_RM
    AI4NEB_INSTALLED = True
except:
    AI4NEB_INSTALLED = False

#%%
l_dic = {'4641.0' : ('N', 3, '4641A', 1),
         '4651.0' : ('O', 2, '4649.13A', 1),
         '4659.0' : ('Fe', 3, '4659A', 0),
         '4662.0' : ('O', 2, '4661.63A', 1),
         '4686.0' : ('He', 2, '4686A', 1),
         '4711.0' : ('Ar', 4, '4711A', 0),
         '4713.0' : ('He', 1, '4713A', 1),
         '4741.0' : ('Ar', 4, '4740A', 0),
         '4861.0' : ('H', 1, '4861A', 1),
         '4959.0' : ('O', 3, '4959A', 0),
         '5200.0' : ('N', 1, '5200A', 0),
         '5343.0' : ('C', 2, '5342.0A', 1),
         '5519.0' : ('Cl', 3, '5518A', 0),
         '5539' : ('Cl', 3, '5538A', 0),
         '5677' : ('N', 2, '5676.02A', 1),
         '5681.0' : ('N', 2, '5679.56A', 1),
         '5755.0' : ('N', 2, '5755A', 0),
         '5877' : ('He', 1, '5876A', 1),
         '6301.0' : ('O', 1, '6300A', 0),
         '6313' : ('S', 3, '6312A', 0),
         '6365.0' : ('O', 1, '6364A', 0),
         #'6458.0' : ('C', 2, '6458A', 1), # Sure of this?
         '6463.0' : ('C', 2, '6462.0A', 1),
         '6549.0' : ('N', 2, '6548A', 0),
         '6564.0' : ('H', 1, '6563A', 1),
         '6585.0' : ('N', 2, '6584A', 0),
         '6679.0' : ('He', 1, '6678A', 1),
         '6719.0' : ('S', 2, '6716A', 0),
         '6733.0' : ('S', 2, '6731A', 0),
         #'7002' : ('O', 4, '7004A', 0),
         '7006.0' : ('Ar', 5, '7005A', 0),
         '7067.0' : ('He', 1, '7065A', 1),
         '7137.0' : ('Ar', 3, '7136A', 0),
         '7283' : ('He', 1, '7281A', 1),
         # 7313???
         '7321.0' : ('O', 2, '7320A', 0),
         '7332.0' : ('O', 2, '7330A', 0),
         '7532.0' : ('Cl', 4, '7531A', 0),
         '7753.0' : ('Ar', 3, '7751A', 0),
         '7772.0' : ('O', 1, '7771A', 1),
         '7774.0' : ('O', 1, '7773A', 1),
         '7777' : ('O', 1, '7775A', 1),
         '8048.0' : ('Cl', 4, '8046A', 0),
         '8730.0' : ('C', 1, '8728A', 0),
         #'8736.0' : ('He', 1, '8733A', 1),
         '8753.0' : ('H', 1, '8750A', 1), 
         '9071.0' : ('S', 3, '9069A', 0),
         '9229.0' : ('H', 1, '9229A', 1)
         }


#%%

def rename_files(data_dir, name, newname=None):
    
    """
    rename_files(data_dir = '/Users/christophemorisset/DATA/MUSE_Jorge/hektor_fits/ngc6778_maps/NGC6778_MUSEstat_maps/maps', 
                 name = 'NGC6778_MUSE_', newname='NGC6778_MUSE_b_')
    """
    if newname is None:
        newname = name
    for lam_str in l_dic:
        l = l_dic[lam_str]
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

#%%
class PipeLine(object):
    
    def __init__(self, 
                 data_dir, 
                 name,
                 error_str='_error', 
                 err_default=0.0,
                 PDF_name='fig'):
        """


        Returns
        -------
        None.

        """
        self.log_ = pn.log_
        
        self.data_dir = data_dir
        self.name = name
        self.error_str = error_str
        self.err_default = err_default
        self.MC_done = False
        self.N_MC = None
        self.PDF_name = PDF_name
        self.TeNe = {}
        self.NII_corrected = False
        self.OII_corrected = False
        
        self.load_obs()
        """
        self.add_MC(N_MC)
        self.red_cor_obs()
        """
        
    def load_obs_deprecated(self):
        
        self.obs = pn.Observation(corrected = False)
        for lam_str in l_dic:
            l = l_dic[lam_str]
            data_file = self.data_dir / Path('{}{}.fits'.format(self.name, lam_str))
            fits_hdu = fits.open(data_file)[0]
            fits_data = fits_hdu.data
            err_data_file = self.data_dir / Path('{}{}{}.fits'.format(self.name, lam_str, self.error_str))
            err_fits_hdu = fits.open(err_data_file)[0]
            err_fits_data = err_fits_hdu.data
            if l[3] == 1:
                perm = 'r'
            else:
                perm = ''
            label='{}{}{}_{}'.format(l[0], l[1], perm, l[2])
            line = pn.EmissionLine(label=label,
                                   obsIntens=fits_data.ravel(), 
                                   obsError=err_fits_data.ravel(), 
                                   corrected=False, errIsRelative=False)
            self.obs.addLine(line)
        self.wcs = WCS(fits_hdu.header).celestial
        self.data_shape = fits_data.shape
        self.n_obs = self.obs.n_obs
        self.obs.names = ['N{}N'.format(i) for i in range(self.n_obs)]
        self.n_lines = self.obs.n_lines
        
    def load_obs(self):
        
        obs_name = Path(self.data_dir) / Path(self.name)
        self.obs = pn.Observation(obs_name, fileFormat='fits_IFU', 
                                  corrected = False, 
                                  errStr=self.error_str, 
                                  errIsRelative=False,
                                  err_default=self.err_default,
                                  addErrDefault = True)
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
                return self.get_image(label=label[0], type_=type_) / self.get_image(label=label[1], type_=type_)
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
            with np.errstate(divide='ignore'):
                this_image = np.log10(this_image)
        im=ax.imshow(this_image, interpolation=interpolation, **kwargs)
        cb = f.colorbar(im, ax=ax)
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
        
        
    def plot_SN(self, ax=None, data=None, label=None, title=None, **kwargs ):

        if ax is None:
            f, ax = plt.subplots(subplot_kw={'projection': self.obs.wcs}, figsize=(8,8))
        else:
            f = plt.gcf()
        med = self.get_image(data=data, label=label, type_='median')
        std = self.get_image(data=data, label=label, type_='std')
        with np.errstate(divide='ignore'):
            this_image = med / std
        im=ax.imshow(this_image, **kwargs)
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
        R_He = self.obs.getIntens()['He1r_7281A'] / self.obs.getIntens()['He1r_6678A']
        
        Te = alphas * R_He + betas
        Te[np.isinf(Te)] = np.nan
        
        self.TeNe['He1']['Te'] = Te
            

    def add_Te_BJ(self, den=1e3, Hep=0.09, Hepp = 0.01):
    
        cont = pn.Continuum()
        tab_tem = np.linspace(1000, 30000, 100)
        tab_den = np.ones_like(tab_tem) * den
        tab_Hep = np.ones_like(tab_tem) * Hep
        tab_Hepp = np.ones_like(tab_tem) * Hepp
    
        tab_BJ =  cont.BJ_HI(tab_tem, tab_den, tab_Hep, tab_Hepp, wl_bbj = 8100, wl_abj = 8400, HI_label='9_3')
        tem_inter = interp1d(tab_BJ, tab_tem, bounds_error=False)
    
        self.TeNe['BJ'] = {}
        C_8100 = self.obs.getIntens()['H1r_8100.0']
        C_8400 = self.obs.getIntens()['H1r_8400.0']
        HI = self.obs.getIntens()['H1r_9229A']
    
        BJ_HI = (C_8100 - C_8400) /  HI
        self.TeNe['BJ']['Te'] = tem_inter(BJ_HI)
        
        #self.TeNe['BJ']['Te'] = cont.T_BJ(BJ_HI, den, He1_H, He2_H, wl_bbj = 8100, wl_abj = 8400, HI_label='9_3',
        #                              T_min=5e2, T_max=3e4)
        return BJ_HI
        
        
    def set_abunds(self, IP_cut = 35, label=None):
        
        atom_dic = {}
        self.abund_dic = {}
        Hbeta = self.obs.getIntens()['H1r_4861A']
        
        
        for line in self.obs.getSortedLines():
            if label is None or line.label == label: 
                if line.atom not in atom_dic:
                    if line.atom[-1] == 'r':
                        atom = pn.RecAtom(line.elem, line.spec, case='A')
                        IP = pn.utils.physics.IP[atom.elem][atom.spec-1]
                    else:
                        atom = pn.Atom(line.elem, line.spec)
                        if atom.spec-2 < 0:
                            IP = 0.
                        else:
                            IP = pn.utils.physics.IP[atom.elem][atom.spec-2]
                    atom_dic[line.atom] = (atom, IP)
                else:
                    atom, IP = atom_dic[line.atom]
                
                if IP < IP_cut:
                    Te = self.TeNe['N2S2']['Te']
                    Ne = self.TeNe['N2S2']['Ne']
                else:
                    Te = self.TeNe['S3S2']['Te']
                    Ne = self.TeNe['S3S2']['Ne']
                self.log_.message('Abund from {} done.'.format(line.label))
                self.abund_dic[line.label] = atom.getIonAbundance(line.corrIntens/Hbeta, Te, Ne, to_eval=line.to_eval, Hbeta=1.)
        
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
        
        I_7320 = self.obs.getIntens()['O2_7320A']
        I_7330 = self.obs.getIntens()['O2_7330A']
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
        R_7325_REC = emisP / emisR
        
        I_7325_new = I_7325 - R_7325_REC * I_REC

        for line in self.obs.lines:
            if line.label == 'O2_7320A':
                line.corrIntens = I_7325_new * I_7320 / I_7325
            if line.label == 'O2_7330A':
                line.corrIntens = I_7325_new * I_7330 / I_7325
        self.OII_corrected = True

        
#%%



