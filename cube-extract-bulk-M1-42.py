#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 08:47:38 2020

@author: hmonteiro
"""

import astropy
import pyspeckit
import pylab as pl
import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import sys
from astropy import units as u
import spectral_cube
from matplotlib.colors import LogNorm
from astropy.io import fits
import os
import pickle
from scipy import ndimage

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
plt.close('all')

data_dir = os.getenv("HOME")+'/Google Drive/work/PN/MUSE_Jorge/data/'
maps_dir = os.getenv("HOME")+'/Google Drive/work/PN/MUSE_Jorge/maps/'

#load 1:1 cube
cube = pyspeckit.Cube(data_dir+'M142_MUSE_LONG.fits', kwargs={'hdu': 1})
cube_er = pyspeckit.Cube(data_dir+'M142_MUSE_LONG_ER.fits', kwargs={'hdu': 1})

scl_fac = 1

lines_tofit = [
    #[['NIII4634','OII4641','OII4649', 'OII4661' ],[4632., 4640., 4648., 4660.]],              
    [['HeII4686' ],[4684.]],
    [['[ArIV]4711','HeI4713' ],[4710.,4711.]],
    [['[ArIV]4740' ],[4739.]],
    [[r'$H\beta$' ],[4860.]],
    [['[OIII]4959' ],[4958.]],
    [['[NI]5197' ],[5198.]],
    [['l5339','CII5342' ],[5339,5343.]],
    [['[ClIII]5518', '[ClIII]5538'],[5516.,5536.]],  
    [['NII5677', 'NII5679' ],[5677, 5681.]], 
    [['[NII]5755'],[5753.]],
    [['HeI5876' ],[5874]],
    [['[SIII]6312' ],[6310]],
    [['[OI]6300' ],[6299.]],
    [['[OI]6363' ],[6362.]],
    [['CII6462' ],[6460.]],
    [['[NII]6548'],[6546.]],
    [[r'$H\alpha$' ],[6561.]],  
    [['[NII]6584'],[6582.]],   
    [['HeI6678'],[6677.]],
    [['[SII]6718', '[SII]6732'],[6715.,6729.]],
    [['[ArV]7005'],[7006.]],
    [['HeI7065'],[7063.]],
    [['[ArIII]7136'],[7134.]],
    [['He I 7281'],[7280]],
    [['[OII]7318+19'],[7318.]],      
    [['[OII]7329+30'],[7328.]],
    [['[ClIV]7530'],[7528.]],
    [['[ArIII]7751'],[7749.]], 
    [['OI7771' , 'OI7773', 'OI7774'],[7770., 7772., 7773]],
    [['[ClIV]8045'],[8044.]],
    [['l?','[FeIII]8729'],[8725.,8731.]],
    [['HI8751'],[8748.]],    
    [['[SIII]9068'],[9067.]],
    [['HI9229'],[9227.]],
                ]
               

# lines_tofit = [ [['NIII4634','OII4641','OII4649', 'OII4661' ],[4632., 4640., 4648., 4660.]] ]              
# lines_tofit = [ [['HeII4686' ],[4684.]] ]
# lines_tofit = [ [['[ArIV]4711','HeI4713' ],[4710.,4711.]] ]
# lines_tofit = [ [['[ArIV]4740' ],[4739.]] ]#cube = pyspeckit.Cube(data_dir+'M142_MUSE_120s_scaled.fits', kwargs={'hdu': 1})

# lines_tofit = [ [[r'$H\beta$' ],[4860.]] ]
# lines_tofit = [ [['[OIII]4959' ],[4958.]] ]
# lines_tofit = [ [['[NI]5197' ],[5198.]] ]
# lines_tofit = [ [['l5339','CII5342' ],[5339,5343.]]  ]
# lines_tofit = [ [['[ClIII]5518', '[ClIII]5538'],[5516.,5536.]] ]  
# lines_tofit = [ [['NII5677', 'NII5679' ],[5675, 5678.]] ] 
# lines_tofit = [ [['[NII]5755'],[5753.]] ]
# lines_tofit = [ [['HeI5876' ],[5874]] ]
# lines_tofit = [ [['[SIII]6312' ],[6310]] ]
# lines_tofit = [ [['[OI]6300' ],[6299.]] ]
# lines_tofit = [ [['[OI]6363' ],[6362.]] ]
# lines_tofit = [ [['CII6462' ],[6460.]] ]
# lines_tofit = [ [['[NII]6548'],[6546.]] ]
# lines_tofit = [ [[r'$H\alpha$' ],[6561.]]  ]  
# lines_tofit = [ [['[NII]6584'],[6582.]]  ]   
# lines_tofit = [ [['HeI6678'],[6677.]] ]
# lines_tofit = [ [['[SII]6718', '[SII]6732'],[6715.,6729.]] ]
# lines_tofit = [ [['[ArV]7005'],[7006.]] ]
# lines_tofit = [ [['HeI7065'],[7063.]] ]
# lines_tofit = [ [['[ArIII]7136'],[7134.]] ]
# lines_tofit = [ [['[OII]7318+19'],[7318.]] ]      
# lines_tofit = [ [['[OII]7329+30'],[7328.]] ]
# lines_tofit = [ [['[ClIV]7530'],[7528.]] ]
# lines_tofit = [ [['[ArIII]7751'],[7749.]] ]  
# lines_tofit = [ [['OI7771' , 'OI7773', 'OI7774'],[7770., 7772., 7773]] ]
# lines_tofit = [ [['[ClIV]8045'],[8044.]] ]
# lines_tofit = [ [['l?','[FeIII]8729'],[8725.,8731.]] ]
# lines_tofit = [ [['HI8751'],[8748.]] ]    
# lines_tofit = [ [['[SIII]9068'],[9067.]] ]
# lines_tofit = [ [['HI9229'],[9227.]] ]
# lines_tofit = [ [['He I 7281'],[7280]]  ]

print('A            A_er      x0           x0_er     sig       sig_er')
int_fluxes = []
for item in lines_tofit:
    
    lines_id, lines = item[0],item[1]
        
    dlines = 7.
    lines_er = 3.
    lwidth = 1.2
    lwidth_er = 0.25
    
    # Slice the cube over the wavelength range you'd like to fit
    cube_fit = cube.slice(np.min(lines)-dlines,np.max(lines)+dlines,unit='Angstrom')
    cube_er_fit = cube_er.slice(np.min(lines)-dlines,np.max(lines)+dlines,unit='Angstrom')

    # define guesses to be used
    
    guesses = []
    fix_par = []
    lim_par = []
    par_lim = []
    int_par_lim = []
    int_guess = []
    
    # set up CASA-like shortcuts
    F=False; T=True    #cube_fit = cube.slice(np.min(lines)-dlines,np.max(lines)+dlines,unit='Angstrom')

    
    for line in lines:
        
        Nspaxel = cube_fit.cube.shape[1]*cube_fit.cube.shape[2]
        
        line_int = np.where(np.abs(cube_fit.xarr - line*u.AA) < lines_er*u.AA)
        
        ang_cube = np.tile(cube_fit.xarr[line_int],(cube_fit.cube.shape[2],cube_fit.cube.shape[1],1)).T.value
        ang_map = np.nansum(cube_fit.cube[line_int]*ang_cube,axis=0)/np.nansum(cube_fit.cube[line_int],axis=0)
    
        base1 = np.nanmedian(cube_fit.cube[np.where(cube_fit.xarr < (np.min(lines)-3*lwidth/2)*u.AA)[0],:,:],axis=0)
        base2 = np.nanmedian(cube_fit.cube[np.where(cube_fit.xarr > (np.max(lines)+3*lwidth/2)*u.AA)[0],:,:],axis=0)  
        base = (base1 + base2)/2.
        
        temp_cube = cube_fit.cube - base[None,:,:]
        noise1 = np.nanstd(temp_cube[np.where(cube_fit.xarr < (np.min(lines)-3*lwidth/2)*u.AA)[0],:,:],axis=0)
        noise2 = np.nanstd(temp_cube[np.where(cube_fit.xarr > (np.max(lines)+3*lwidth/2)*u.AA)[0],:,:],axis=0)
        noise_map = (noise1+noise2)/2
        
        noise_map_est = np.nanmean(np.abs(cube_fit.cube - ndimage.median_filter(cube_fit.cube,[3,1,1])),axis=0)
        noise_map = np.nanmean(np.sqrt(cube_er_fit.cube),axis=0)

        amp = np.nanmax(cube_fit.cube[np.where(np.abs(cube_fit.xarr - line*u.AA) < lwidth*u.AA)[0],:,:],axis=0) - base
        amp[amp < 0.] = 0.
        
        l0 = ang_map
        lsig = amp*0. + lwidth
        
        if (line == lines[0]):
            fix_par = fix_par + [F, F, F, F]
            lim_par = lim_par + [(T,T), (T,T), (T,T),(T,T)] 
            par_lim = par_lim + [(0.,np.nanmax(cube_fit.cube)), (0.,np.nanmax(cube_fit.cube)), (line-lines_er,line+lines_er),(0.8,lwidth+lwidth_er)] 
            int_par_lim = int_par_lim + [(0.,np.nanmax(cube_fit.cube)*Nspaxel), (0.,np.nanmax(cube_fit.cube)*Nspaxel), (line-lines_er,line+lines_er),(0.8,lwidth+lwidth_er)] 
            base[base < 0.] = 0.
            l0[l0 < (line-lines_er)] = line-lines_er
            l0[l0 > (line+lines_er)] = line+lines_er
            guesses = guesses + [base,amp,l0,lsig]
            int_guess = int_guess + [np.nansum(base),np.nansum(amp),np.nanmean(l0),np.nanmean(lsig)]
        else:
            fix_par = fix_par + [T, F, F, F]
            lim_par =  lim_par + [(T,T), (T,T), (T,T),(T,T)]
            par_lim = par_lim + [(0.,0.), (0.,np.nanmax(cube_fit.cube)), (line-lines_er,line+lines_er),(0.8,lwidth+lwidth_er)]
            int_par_lim = int_par_lim + [(0.,0.), (0.,np.nanmax(cube_fit.cube)*Nspaxel), (line-lines_er,line+lines_er),(0.8,lwidth+lwidth_er)] 
            l0[l0 < (line-lines_er)] = line-lines_er
            l0[l0 > (line+lines_er)] = line+lines_er
            guesses = guesses + [amp*0.,amp,l0,lsig]
            int_guess = int_guess + [0.,np.nansum(amp),np.nanmean(l0),np.nanmean(lsig)]
            
    # Do an initial plot & fit of a single spectrum
    # at a pixel with good S/N
    sp = pyspeckit.Spectrum(data=cube_fit.cube.sum(axis=1).sum(axis=1), error=np.sqrt(cube_er_fit.cube.sum(axis=1).sum(axis=1))
                            ,xarr=cube_fit.xarr)
    sp.plotter()
    
    sp.specfit(fittype='vheightgaussian',guesses=int_guess,quiet=True,save=False,
               fixed=fix_par, parlimited=lim_par,parlimits=int_par_lim,)               
    #sp.specfit.plot_components(add_baseline=True)
    #sp.specfit.plotresiduals()
    plt.savefig(maps_dir+'M142_MUSE_'+str(lines).replace(" ", "")+'.png')
    
    cont=2

    for i in range(len(lines)):
        A,x0,sig = sp.specfit.modelpars[4*i+1:4*i+4]
        A_er,x0_er,sig_er = sp.specfit.modelerrs[4*i+1:4*i+4]
        
        int_flux = A*sig*np.sqrt(2.*np.pi)
        int_flux_er = int_flux * np.sqrt((A_er/A)**2 + (sig_er/sig)**2)
        
        int_fluxes.append([A,A_er,x0,x0_er,sig,sig_er])
        
        try:
            print('%8.3f  %8.3e  %8.3e'%(x0,int_flux,int_flux_er))
        except:
            print('line ',lines[i], ' has no fit...')
    
    #plt.savefig()
    # sys.exit()
    continue

    
    ## Here's where all the fitting happens.
    ## With the "parlimited" and "parlimits" keywords, I have restricted
    ## the range for the wavelength centroid and linewidth parameters.
    ## With the "fixed" keyword, I have held the 2nd gaussian's background level
    ## to zero, and the "signal_cut" keyword rejects fits for voxels below a
    ## user-specified S/N threshold.
    
    cube_fit.fiteach(use_nearest_as_guess=False,
                     use_neighbor_as_guess=True,
                     # blank_value = -999
                     #use_lmfit=True,
                     guesses=np.array(guesses),
                     fittype='vheightgaussian',
                     #integral=True,
                     multicore=7,
                     negamp=False,
                     verbose=False,
                     verbose_level=2,
                     continuum_map = base,
                     errmap=noise_map,                     
                     parlimited=lim_par,
                     parlimits=par_lim,
                     fixed=fix_par,
                     signal_cut=0.1,
                     show_components=True,
                     skip_failed_fits = True,
                     start_from_point=(int(70/scl_fac),int(70/scl_fac)))
    
        
    ##############################################################################
    # define map header
    from astropy import wcs
    map_header = wcs.WCS(cube.header).sub([wcs.WCSSUB_CELESTIAL]).to_header()
    
    residual_cube = cube_fit.cube - cube_fit.get_modelcube()
    
    hasfit = np.zeros(cube_fit.has_fit.shape)
    hasfit[np.where(cube_fit.has_fit == True)] = 1.
    hdu = fits.PrimaryHDU(hasfit)
    hdulist = fits.HDUList([hdu])
    hdulist[0].header = map_header    
    # hdulist[0].header['WCSAXES'] = 2   
    hdulist.writeto(maps_dir+'M142_MUSE_'+str(lines).replace(" ", "")+'_has_fit.fits',overwrite=True, output_verify='fix')


    # save fit results to file
    cube_fit.write_fit(maps_dir+'M142_MUSE_'+str(lines).replace(" ", "")+'.fits', overwrite=True)
    
    cont=2
    for i in range(len(lines)):
        
        par = cube_fit.parcube[4*i+2,:,:]
        p_max = np.nanmax(par)
        p_min = np.nanmin(par[par > 0.])
        cube_fit.mapplot.figure=pl.figure(cont)
        cube_fit.mapplot(estimator=4*i+2,vmax=p_max,vmin=p_min)
        cube_fit.mapplot.axis.set_title(str(lines[i])+" Line Center")
        
        cube_fit.mapplot.figure=pl.figure(cont+1)
        amp_max = np.nanmax(cube_fit.parcube[4*i+1,:,:])
        amp_min = np.nanmin(cube_fit.parcube[4*i+1,:,:])
        print(amp_max)
        cube_fit.mapplot(estimator=4*i+1,vmax=amp_max,vmin=amp_min, plotkwargs={'stretch':'arcsinh','vmid':1000})
        cube_fit.mapplot.axis.set_title(str(lines[i]))
        
        flux_map = cube_fit.parcube[4*i+1,:,:]*cube_fit.parcube[4*i+3,:,:]*np.sqrt(2.*np.pi)
        flux_map_err = flux_map * np.sqrt( (cube_fit.errcube[4*i+1,:,:]/cube_fit.parcube[4*i+1,:,:])**2 + 
                                             (cube_fit.errcube[4*i+3,:,:]/cube_fit.parcube[4*i+3,:,:])**2 ) 
                
        
        # save maps
        hdu = fits.PrimaryHDU(flux_map)
        hdulist = fits.HDUList([hdu])
        hdulist[0].header = map_header    
        # hdulist[0].header['WCSAXES'] = 2   
        hdulist.writeto(maps_dir+'M142_MUSE_'+str(lines[i])+'.fits',overwrite=True, output_verify='fix')
    
        hdu = fits.PrimaryHDU(flux_map_err)
        hdulist = fits.HDUList([hdu])
        hdulist[0].header = map_header    
        # hdulist[0].header['WCSAXES'] = 2   
        hdulist.writeto(maps_dir+'M142_MUSE_'+str(lines[i])+'_error.fits',overwrite=True, output_verify='fix')
    
        # estimate error map as ALFA
        FWHM = 2.3548 * cube_fit.parcube[4*i+3,:,:]
        peak = cube_fit.parcube[4*i+1,:,:] 
        peak_er = cube_fit.errcube[4*i+1,:,:]
        peak_er_s = ndimage.median_filter(peak_er,6)
        ind_out = np.abs(peak_er - peak_er_s) > 3.*np.nanstd(np.abs(peak_er - peak_er_s))
        peak_er[ind_out] = peak_er_s[ind_out]
        SN = 0.67 * (peak/np.nanstd(residual_cube,axis=0)) * np.sqrt(FWHM/cube_fit.header['CDELT3'])
        alfa_er = flux_map/SN
    
        hdu = fits.PrimaryHDU(alfa_er)
        hdulist = fits.HDUList([hdu])
        hdulist[0].header = map_header    
        hdulist.writeto(maps_dir+'M142_MUSE_'+str(lines[i])+'_error_alfalike.fits',overwrite=True, output_verify='fix')
    
        cont += 2
                
    plt.close('all')

    











