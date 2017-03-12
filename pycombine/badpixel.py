import numpy as np
from astropy.stats import sigma_clip
from misc import read_fits


def make_bpm(flatdir,sigma=3):
    '''Assuming the flats are sorted. E.g. in increasing or decreasing intensity. Then clip the
    Values which are of. Using for loops. So slow. Returns a matrix which is 1 whre a bp is found.'''
    print('Making BPM using flats')
    fns,flats,header = read_fits(flatdir)
    bpm = np.full(flats.shape[1:],0).astype(np.int16)
    gradmap = np.full(flats.shape[1:],np.nan)
    print('Sorting the flats')
    nflats = flats.shape[0]
    sums = []
    for ii in range(nflats):
        sums.append(np.sum(flats[ii,::]))
    sumsorted = np.argsort(sums)
    flats = flats[sumsorted,::]
    iflats = np.linspace(0,nflats-1,nflats).astype(int)
    for yy in range(flats.shape[1]):
        if yy%10 ==0: print('At row %d of %d'%(yy,flats.shape[1]))
        for xx in range(flats.shape[2]):
            values = sigma_clip(flats[:,yy,xx],sig=5) #filter cosmic rays
            gradmap[yy,xx] = np.polyfit(iflats[~values.mask],\
                                        values[~values.mask],deg=1)[1]
    bpm[sigma_clip(gradmap,sigma=sigma).mask] = 1
    return bpm
    
    
