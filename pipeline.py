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

from pathlib import Path

#%%

class PipeLine(object):
    
    def __init__(self, data_dir=None):
        """


        Returns
        -------
        None.

        """
        self.log_ = log_
        
        self.data_dir = data_dir

        pass

    def load_obs(self):
        pass
        
    
    
        
#%%
#if __name__ == "__main__":
    
data_dir = Path('/Users/christophemorisset/DATA/MUSE_Jorge/hektor_fits/ngc6778_maps/ngc6778_long_maps')

lines_dict = {1:'4641.0',2:'4651.0',3:'4659.0', 4:'4662.0',5:'4686.0',6:'4711.0',7:'4713.0',8:'4741.0',
              9:'4861.0',10:'4959.0',11:'5200.0', 12:'5343.0',13:'5677',14:'5681.0',15:'5755.0',16:'5877',
              17:'6301.0',18:'6313.0',19:'6365.0',20:'6463.0', 21:'6549.0',22:'6564.0',23:'6585.0',24:'6679.0',
              25:'6719.0',26:'6733.0',27:'7002',28:'7006.0',29:'7067.0',30:'7137.0', 31:'7321.0',32:'7332.0',
              33:'7532.0',34:'7753.0',35:'7772.0',36:'7774.0',37:'7777',38:'8048.0',39:'8730.0', 40:'8736.0',
              41:'8753.0',42:'9071.0',43:'9072.0'}
id_dict = {1:'NIII_4641',2:'OII_4649',3:'FeIII_4659', 4:'OII_4662',5:'HeII_4686',6:'ArIV_4711',7:'HeI_4713',8:'ArIV_4740',
           9:'HI_4861',10:'OIII_4959',11:'NI_5200', 12:'CII_5342',13:'NII_5676',14:'NII_5679',15:'NII_5755',16:'HeI_5876',
           17:'OI_6300',18:'SIII_6312',19:'OI_6363',20:'CII_6461',21:'NII_6548',22:'HI_6562',23:'NII_6583',24:'HeI_6678',
           25:'SII_6717',26:'SII_6730',27:'OIV_7004',28:'NeV_7005',29:'HeI_7065',30:'ArIII_7135',31:'OII_7320',32:'OII_7330',
           33:'ClIV_7530',34:'ArIII_7751',35:'OI_7771',36:'OI_7773',37:'OI_7775',38:'ClIV_8046',39:'CI_8727', 40:'HeI_8733',
           41:'HI_8750',42:'SIII_9068'}

l_dic = {'4641.0': ('N', 2, '4641A'),
         '4651.0' : ('O', 2, '4649A'),
         '4659.0' : ('Fe', 3, '4659A'),
         '4662.0' : ('O', 2, '4662A'),
         '6549.0' : ('N', 2, '6548A'),
         '6585.0' : ('N', 2, '6584A'),
         '5755.0' : ('N', 2, '5755A'),
         '6719.0' : ('S', 2, '6716A'),
         '6733.0' : ('S', 2, '6731A'),
         '4959.0' : ('O', 3, '4959A')
         }
#%%
obs = pn.Observation()
for lam_str in l_dic:
    data_file = data_dir / Path('ngc6778_MUSE_{}.fits'.format(lam_str))
    fits_hdu = fits.open(data_file)[0]
    fits_data = fits_hdu.data
    err_data_file = data_dir / Path('ngc6778_MUSE_{}_error.fits'.format(lam_str))
    err_fits_hdu = fits.open(err_data_file)[0]
    err_fits_data = err_fits_hdu.data
    
    line = pn.EmissionLine(l_dic[lam_str][0], l_dic[lam_str][1], l_dic[lam_str][2], 
                           obsIntens=fits_data.ravel(), 
                           obsError=err_fits_data.ravel(), corrected=True)
    obs.addLine(line)
wcs = WCS(fits_hdu.header).celestial
#%% 
diags = pn.Diagnostics()
diags.ANN_inst_kwargs['verbose']=True
diags.addDiagsFromObs(obs)
#%%
pn.log_.timer('Starting', quiet=True)
Te, Ne = diags.getCrossTemDen('[NII] 5755/6584', '[SII] 6731/6716', obs=obs)
pn.log_.timer('Paralelized getCrossTemDen done')
#%%
pn.log_.timer('Starting', quiet=True)
pn.config.verbose=2
Te_ANN, Ne_ANN = diags.getCrossTemDen('[NII] 5755/6584', '[SII] 6731/6716', obs=obs, use_ANN=True)
pn.log_.timer('ANN getCrossTemDen done')

#%%
f, (ax1, ax2) = plt.subplots(2,1,figsize=(10,15))
ax1.scatter(Te, Te-Te_ANN, alpha=0.01)
ax1.set_xlim(0, 20000)
ax1.set_ylim(-1000,1000)

ax2.scatter(Ne, Ne-Ne_ANN, alpha=0.01)
ax2.set_xlim(0, 1000)
ax2.set_ylim(-100,100)
#%%
Te = np.reshape(Te_ANN, fits_data.shape)
f, ax = plt.subplots(subplot_kw={'projection': wcs})
im = ax.imshow(Te, vmin=7000, vmax=13000, cmap='jet')
cb = f.colorbar(im, ax=ax)




