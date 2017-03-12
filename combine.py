import os
from astropy.io import fits
from astropy import wcs
from astropy import units as u
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import register_translation
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.interpolation import shift
from pycombine.params import *
from pycombine import misc,fix_pix,data,darkflat,cut_star,badpixel,align
from PIL import Image

import ipdb

col = {'R':'HA','G':'V','B':'B'}

def do(waves =['HA','B','V','R'],target=None,darkbias ='bias',pardir = None, datadir =None,darkdir=None,flatdir=None,ncpu=2,date=None,keepfrac=1,cut_range=False):
    '''A Pipeline to reduce 70cm data. (modified NICI version) Provide darks in darkdir, flats in flatdir.
    Assuming dithering, so taking the neighbouring two images with a different dither
    position and interpolating between the pixels to to get the flat.
    BP: Using flats for BP detection: Values changing differently than the others are
    being replaced by median of good neighbours. Taken from Jordans pipeline.'''
    for waveband in waves:
        print('Processing waveband %s'%waveband)
        wavebands = ['I','R','V','B','HA']
        waveband = waveband.upper()
        if waveband not in wavebands: raise ValueError('Your waveband %s not known. Choose from %s'\
                                                       %(waveband,wavebands))
        if target == None: target = 'ORION_NEBEL'
        if date   == None: date ='20170310'
        if pardir == None: pardir  = '/windows/Users/Stefan/Pictures/2015+_Astrobilder/70cm/'+date+'/'
        if datadir== None: datadir = pardir+target+'/'
        if darkbias == 'bias':
            if darkdir== None: darkdir = pardir+'biases/'
        else:
            if darkdir== None: darkdir = pardir+'darks/'
        if flatdir== None: flatdir = pardir+'flats/'+waveband+'/'
        intermdir = datadir+'interm/'
        finaldir =  datadir+'results/'
        for direct in [finaldir,intermdir]:
            if not os.path.exists(direct):
                os.mkdir(direct)
        fnbpm = 'bpm.fit'

        #getting and sorting darks and flats
        ddata,dTexp = darkflat.sort(darkdir,hexpt)
        fdata,fTexp = darkflat.sort(flatdir,hexpt)
        fdata = darkflat.subtract_dark(fdata,fTexp,ddata,dTexp)
    
        #making the bpm
        
        if os.path.exists(pardir+fnbpm):
            print('Found bpm in %s'%(pardir+fnbpm))
            bpm = fits.getdata(pardir+fnbpm)
        else:
            print('No bpm found. Making it')
            bpm = badpixel.make_bpm(flatdir)
            fits.writeto(pardir+fnbpm,bpm)

        #getting science data
        fns,sdata,heads = misc.read_fits(datadir,ending='_'+waveband+'.fit')
        sdata = darkflat.subtract_dark(sdata,[heads[0][hexpt]],ddata,dTexp)
        sdata = darkflat.divide_flat(sdata,[heads[0][hexpt]],fdata,fTexp)
        #fixpixing
        print('Creating fixpix map')
        bad_and_neighbors = fix_pix.find_neighbors(bpm,n=4)
        for iim in range(sdata.shape[-3]):
            fix_pix.correct_with_precomputed_neighbors(sdata[iim,::],bad_and_neighbors)#inplace
        print('Aligning')
        intermdata=np.median(align.align_cut(sdata,imuse='median',cut=False,\
                                             keepfrac=1.,ncpu=ncpu),axis=0)
        sdata = np.median(align.align_cut(sdata,imuse='median',cut=True,\
                                          keepfrac=1.,ncpu=ncpu),axis=0)
        
        
        fits.writeto(intermdir+target+'_comb_'+waveband+'.fit',\
                     intermdata,header=heads[0],\
                     clobber=True)
        fits.writeto(finaldir+target+'_comb_'+waveband+'.fit',\
                     sdata,header=heads[0],\
                     clobber=True)
        
            
        #finding the star positions and time of observation
        #filetable = data.return_quad(datadir,hexpt,ddata,dTexp,fdata,fTexp,bpm)
        #get the right observation for background subtraction based on positions
        #filetable = data.determine_neighbours(filetable)
        #getting science data. Flatfielding, bkgrnd and fixpixing it and saving in intermdir
        #data.flatfield_bkgrnd(fdata,fTexp,filetable,bpm,intermdir)
        ##############################################
    
        #cut out the star, align them and get the paralactic angles
        #cut_star.align_stars(indir=intermdir,outdir=finaldir,ncpu=ncpu,keepfrac=keepfrac)
    print('Combining colors')
    rim,head = fits.getdata(intermdir+target+'_comb_'+col['R']+'.fit',\
                            header=True)
    gim = fits.getdata(intermdir+target+'_comb_'+col['G']+'.fit')
    bim = fits.getdata(intermdir+target+'_comb_'+col['B']+'.fit')
    colim = np.stack([rim,gim,bim],axis=0)
    for ii in range(colim.shape[0]):
        colim[ii,:,:] -= np.min(colim[ii,:,:])
        colim[ii,:,:] *= (255.999/np.max(colim[ii,:,:]))
    colim =align.align_cut(colim,imuse='median',cut=True,\
                                      keepfrac=1.)

    for ii in range(colim.shape[0]):
        if cut_range:
            colrange = np.max(colim[ii,:,:]) - np.min(colim[ii,::])
            colmin = (np.min(colim[ii,:,:]) + 0.01*colrange)
            colmax = (np.max(colim[ii,:,:]) - 0.5*colrange)
            colim[ii,:,:][colim[ii,:,:] <= colmin ] = colmin
            colim[ii,:,:][colim[ii,:,:] >= colmax ] = colmax

        colim[ii,::] -= np.min(colim[ii,::])
        colim[ii,::] = np.sqrt(colim[ii,::])
        colim[ii,::] *= (255.999/np.max(colim[ii,::]))
    fits.writeto(finaldir+target+'_'+col['R']+col['G']+col['B']+'.fit',\
                 colim,clobber=True)
    img = Image.fromarray(np.swapaxes(np.swapaxes(colim.astype(np.uint8),0,2),1,0))
    img.save(finaldir+target+'_'+col['R']+col['G']+col['B']+'.png')
    print('DONE WITH ALL :)')
    import ipdb;ipdb.set_trace()

