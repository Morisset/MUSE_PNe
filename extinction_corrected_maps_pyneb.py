import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from astropy.visualization import astropy_mpl_style
#plt.style.use(astropy_mpl_style)
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
import pyneb as pn

mpl.rcParams['font.size'] = 20

lines_dict = {1:'4641.0',2:'4651.0',3:'4659.0', 4:'4662.0',5:'4686.0',6:'4711.0',7:'4713.0',8:'4741.0',9:'4861.0',10:'4959.0',11:'5200.0', 12:'5343.0',13:'5677',14:'5681.0',15:'5755.0',16:'5877',17:'6301.0',18:'6313.0',19:'6365.0',20:'6463.0', 21:'6549.0',22:'6564.0',23:'6585.0',24:'6679.0',25:'6719.0',26:'6733.0',27:'7002',28:'7006.0',29:'7067.0',30:'7137.0', 31:'7321.0',32:'7332.0',33:'7532.0',34:'7753.0',35:'7772.0',36:'7774.0',37:'7777',38:'8048.0',39:'8730.0', 40:'8736.0',41:'8753.0',42:'9071.0',43:'9072.0'}
id_dict = {1:'NIII_4641',2:'OII_4649',3:'FeIII_4659', 4:'OII_4662',5:'HeII_4686',6:'ArIV_4711',7:'HeI_4713',8:'ArIV_4740',9:'HI_4861',10:'OIII_4959',11:'NI_5200', 12:'CII_5342',13:'NII_5676',14:'NII_5679',15:'NII_5755',16:'HeI_5876',17:'OI_6300',18:'SIII_6312',19:'OI_6363',20:'CII_6461',21:'NII_6548',22:'HI_6562',23:'NII_6583',24:'HeI_6678',25:'SII_6717',26:'SII_6730',27:'OIV_7004',28:'NeV_7005',29:'HeI_7065',30:'ArIII_7135',31:'OII_7320',32:'OII_7330',33:'ClIV_7530',34:'ArIII_7751',35:'OI_7771',36:'OI_7773',37:'OI_7775',38:'ClIV_8046',39:'CI_8727', 40:'HeI_8733',41:'HI_8750',42:'SIII_9068'}
pyneb_id = {1:'NIII_4641',2:'OII_4649',3:'FeIII_4659', 4:'OII_4662',5:'HeII_4686',6:'ArIV_4711',7:'HeI_4713',8:'ArIV_4740',9:'HI_4861',10:'OIII_4959',11:'NI_5200', 12:'CII_5342',13:'NII_5676',14:'NII_5679',15:'NII_5755',16:'HeI_5876',17:'OI_6300',18:'SIII_6312',19:'OI_6363',20:'CII_6461',21:'NII_6548',22:'HI_6562',23:'NII_6583',24:'HeI_6678',25:'SII_6717',26:'SII_6730',27:'OIV_7004',28:'NeV_7005',29:'HeI_7065',30:'ArIII_7135',31:'OII_7320',32:'OII_7330',33:'ClIV_7530',34:'ArIII_7751',35:'OI_7771',36:'OI_7773',37:'OI_7775',38:'ClIV_8046',39:'CI_8727', 40:'HeI_8733',41:'HI_8750',42:'SIII_9068'}
#pne_dict = {1:'NGC6778',2:'M142_2', 3:'Hf22'}
pne_dict = {1:'ngc6778'}
#dir_dict = {1:'noOI_noOIII',2:'noOI_noOIII',3:'10000'}
#xcenter_dict = {1:74,2:82,3:103}
xcenter_dict = {1:74}
#ycenter_dict = {1:74,2:76,3:100}
ycenter_dict = {1:74}

for pne in pne_dict:


    image_hb = get_pkg_data_filename('hektor_fits/'+pne_dict[pne]+'_maps/'+pne_dict[pne]+'_long_maps/'+pne_dict[pne]+'_MUSE_4861.0.fits')
    hdu_hb = fits.open(image_hb)[0]
    image_data_hb = hdu_hb.data
    image_data_hb = np.nan_to_num(image_data_hb,nan=0,posinf=0,neginf=0)
    wcs = WCS(hdu_hb.header).celestial # Import the WCS header

    image_ha = get_pkg_data_filename('hektor_fits/'+pne_dict[pne]+'_maps/'+pne_dict[pne]+'_long_maps/'+pne_dict[pne]+'_MUSE_6564.0.fits')
    hdu_ha = fits.open(image_ha)[0]
    image_data_ha = hdu_ha.data
    image_data_ha = np.nan_to_num(image_data_ha,nan=0,posinf=0,neginf=0)

    image_hb_alfa = get_pkg_data_filename('hektor_fits/'+pne_dict[pne]+'_maps/'+pne_dict[pne]+'_long_maps/'+pne_dict[pne]+'_TRIMMED_4861.33.fits')
    hdu_hb_alfa = fits.open(image_hb_alfa)[0]
    image_data_hb_alfa = hdu_hb_alfa.data
    image_data_hb_alfa = np.nan_to_num(image_data_hb_alfa,nan=0,posinf=0,neginf=0)

    image_ha_alfa = get_pkg_data_filename('hektor_fits/'+pne_dict[pne]+'_maps/'+pne_dict[pne]+'_long_maps/'+pne_dict[pne]+'_TRIMMED_6562.77.fits')
    hdu_ha_alfa = fits.open(image_ha_alfa)[0]
    image_data_ha_alfa = hdu_ha_alfa.data
    image_data_ha_alfa = np.nan_to_num(image_data_ha_alfa,nan=0,posinf=0,neginf=0)

    ####COMPUTATION OF THE EXTINCTION LAW. HOWARTH 1983 (THE SAME THAN SEATON 1979 IN THE OPTICAL)####

    rc = pn.RedCorr()
    rc.law = 'S79 H83 CCM89'
    rc.setCorr((image_data_ha/image_data_hb) / 2.86,6563.,4861.)
    chb_image=rc.cHbeta
    chb_image=np.nan_to_num(chb_image,nan=0,posinf=0,neginf=0)

    rc.setCorr((image_data_ha_alfa/image_data_hb_alfa) / 2.86,6563.,4861.)
    chb_image_alfa=rc.cHbeta
    chb_image_alfa=np.nan_to_num(chb_image_alfa,nan=0,posinf=0,neginf=0)


    fig = plt.figure(figsize=(11,8))
    fig.add_subplot(111, projection=wcs)
    plt.plot(xcenter_dict[pne],ycenter_dict[pne],'x',ms=8,mew=1,color='black')
    fig=plt.imshow(chb_image, cmap='BuPu')
    plt.title(r'c(H$_{\beta}$) (Hektor)')
    plt.xlabel('Right Ascension')
    plt.ylabel('Declination')
    plt.clim(0.4,1.1)
    cbar = plt.colorbar()
    cbar.ax.get_yaxis().labelpad = 20
    cbar.ax.set_ylabel(r'c(H$\beta$)', rotation=270)
    levels_dict = [0.01*image_data_ha.max(),0.05*image_data_ha.max(),0.1*image_data_ha.max(),0.2*image_data_ha.max(),0.3*image_data_ha.max(),0.4*image_data_ha.max(),0.5*image_data_ha.max(),0.6*image_data_ha.max(),0.7*image_data_ha.max(),0.8*image_data_ha.max(),0.9*image_data_ha.max(),image_data_ha.max()]
    plt.contour(image_data_ha, levels=levels_dict, colors='black', alpha=0.6)
    plt.savefig('hektor_fits/'+pne_dict[pne]+'_maps/linemaps/'+pne_dict[pne]+'_Chb_HaHb_map_wcs.pdf')
    plt.close()

    fig = plt.figure(figsize=(11,8))
    fig.add_subplot(111, projection=wcs)
    plt.plot(xcenter_dict[pne],ycenter_dict[pne],'x',ms=8,mew=1,color='black')
    fig=plt.imshow(chb_image_alfa, cmap='BuPu')
    plt.title(r'c(H$_{\beta}$) (ALFA)')
    plt.xlabel('Right Ascension')
    plt.ylabel('Declination')
    plt.clim(0.4,1.1)
    cbar = plt.colorbar()
    cbar.ax.get_yaxis().labelpad = 20
    cbar.ax.set_ylabel(r'c(H$\beta$)', rotation=270)
    levels_dict = [0.01*image_data_ha_alfa.max(),0.05*image_data_ha_alfa.max(),0.1*image_data_ha_alfa.max(),0.2*image_data_ha_alfa.max(),0.3*image_data_ha_alfa.max(),0.4*image_data_ha_alfa.max(),0.5*image_data_ha_alfa.max(),0.6*image_data_ha_alfa.max(),0.7*image_data_ha_alfa.max(),0.8*image_data_ha_alfa.max(),0.9*image_data_ha_alfa.max(),image_data_ha_alfa.max()]
    plt.contour(image_data_ha_alfa, levels=levels_dict, colors='black', alpha=0.6)
    plt.savefig('hektor_fits/'+pne_dict[pne]+'_maps/linemaps/'+pne_dict[pne]+'_Chb_HaHb_map_wcs_ALFA.pdf')
    plt.close()

    masked_data = np.ma.masked_array(chb_image, chb_image==0) # If no values are zeros
    #masked_data = np.ma.masked_array(chb_image, np.isnan(chb_image)) # If no values are nan
    chb_HaHb_mean = np.ma.mean(masked_data)
    chb_HaHb_std = np.ma.std(masked_data)
    print('The mean c(H$\beta$) from the H$\alpha$/H$\beta$ ratio in %s is %.2f +- %.2f' % (pne_dict[pne],chb_HaHb_mean, chb_HaHb_std))

    masked_data_alfa = np.ma.masked_array(chb_image_alfa, chb_image_alfa==0) # If no values are zeros
    #masked_data = np.ma.masked_array(chb_image, np.isnan(chb_image)) # If no values are nan
    chb_HaHb_mean_alfa = np.ma.mean(masked_data_alfa)
    chb_HaHb_std_alfa = np.ma.std(masked_data_alfa)
    print('The mean c(H$\beta$) from the H$\alpha$/H$\beta$ ratio with ALFA in %s is %.2f +- %.2f' % (pne_dict[pne],chb_HaHb_mean_alfa, chb_HaHb_std_alfa))

plt.figure(figsize=(10,7))
#plt.hist(masked_data.ravel(),bins=np.linspace(0.4,1.1,100))
#plt.hist(masked_data_alfa.ravel(),bins=np.linspace(0.4,1.1,100), alpha=0.5)
plt.hist((image_data_ha/image_data_hb).ravel(),bins=np.linspace(3.5,7,100), alpha=0.7, label='Hektor fit')
plt.hist((image_data_ha_alfa/image_data_hb_alfa).ravel(),bins=np.linspace(3.5,7,100),alpha=0.5, label = 'ALFA fit')
plt.title(r'Histogram of H$\alpha$/H$\beta$ values')
plt.axvline(4.827238214234225,color='g', label = 'collapsed')
plt.xlabel('Value')
plt.ylabel('Number of spaxels')
plt.legend(loc="upper right")
plt.savefig('histogram_HaHb.pdf')

"""
    for line in lines_dict:

        imagen = 'hektor_fits/'+pne_dict[pne]+'maps/'+pne_dict[pne]+'long_maps/'+pne_dict[pne]+'_MUSE_'+lines_dict[line] #limits 0,maxim
        hdu_list = fits.open(imagen+'.fits')
        image_data = hdu_list[0].data
        hdu_list.close()
        
        image_coccor_hb = (image_data/image_data_hb)*10**(chb_image)
        


# PLOT HISTOGRAMS OF C(HB) AND HA/HB RATIO FROM HEKTOR AND ALFA

plt.hist(masked_data.ravel(),bins=np.linspace(0.4,1.1,100))
plt.hist(masked_data_alfa.ravel(),bins=np.linspace(0.4,1.1,100), alpha=0.5)
plt.hist((image_data_ha/image_data_hb).ravel(),bins=np.linspace(3.5,7,100))
plt.hist((image_data_ha_alfa/image_data_hb_alfa).ravel(),bins=np.linspace(3.5,7,100),alpha=0.5)


# COLLAPSE ALL SPECTRA FROM MASKED DATACUBE

cubo = get_pkg_data_filename('NGC6778_DATACUBE_TRIMMED.fits')
hdu = fits.open(cubo)[1]
cubo_masked = np.nan_to_num(hdu.data,nan=0,posinf=0,neginf=0)
test1 = cubo_masked.sum(1).sum(1)
hdu_head = hdu.header
fits.writeto('test1.fits',test1,hdu_head)
"""
