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

def do(waves =['B'],target=None,darkbias ='bias',pardir = None, datadir =None,darkdir=None,flatdir=None,ncpu=1,date=None,keepfrac=1):
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
        if pardir == None: pardir  = os.getcwd()+'/'+date+'/'
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
        fnbpm = 'bpm_orig.fit'
        
        if os.path.exists(pardir+fnbpm):
            print('Found bpm in %s'%(pardir+fnbpm))
            bpm = fits.getdata(pardir+fnbpm)

        fns,flats,heads = misc.read_fits(flatdir)
        
        bad_and_neighbors = fix_pix.find_neighbors(bpm,n=4)
        for iim in range(flats.shape[-3]):
            fix_pix.correct_with_precomputed_neighbors(flats[iim,::],bad_and_neighbors)#inplace
            fits.writeto(flatdir+'new/'+fns[iim],flats[iim,::],header=heads[iim])
        
