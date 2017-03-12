import numpy as np
from multiprocessing import Pool
from scipy.ndimage.interpolation import shift
from scipy.signal import fftconvolve
from gaussfitter import gaussfit


class subreg:
    def __init__(self,reference):
        self.reference=reference
    def __call__(self,im):
        kernel=self.reference[::-1,::-1]
        cor = fftconvolve(im,kernel,mode='same')
        y,x=find_max_star(cor)
        g=gaussfit(cor[max(0, y-40):min(y+40, cor.shape[0]),
                       max(0, x-40):min(x+40, cor.shape[1])])
        shiftx=np.rint(cor.shape[1]/2.) - max(0,x-40)-g[2]
        shifty=np.rint(cor.shape[0]/2.) - max(0,y-40)-g[3]
        #shifts.append((shifty,shiftx))
        return (shifty,shiftx)

    
def bin_median(arr,smaller_by_factor=1,returnStd=False):
    '''bin an array arr by creating super pixels of size
    smaller_by_factor*smaller_by_factor and taking the median. 
    Can optionally also return the standard deviation within
    the super-pixels.
    INPUTS:
    arr: 2d array
    smaller_by_factor: integer
    returnStd: bool, default False
    RETURNS binned_array OR binned_array, bin_std'''
    sub_arrs0=[]
    for i in xrange(smaller_by_factor):
        for j in xrange(smaller_by_factor):
            sub_arrs0.append(arr[i::smaller_by_factor,j::smaller_by_factor])
    sub_arrs=[s[:sub_arrs0[-1].shape[0]-1,:sub_arrs0[-1].shape[1]-1] for s in sub_arrs0]
    if returnStd:
        #zip truncates each sub-arr at the length of the minimum length subarr.
        #this ensures every bin has the same number of datapoints, but throws
        #away data if the last bin doesn't have a full share of datapoints.
        return np.median(sub_arrs,axis=0),np.std(sub_arrs,axis=0)
    else:
        return np.median(sub_arrs,axis=0)


def find_max_star(image):
    '''Median smooth an image and find the max pixel.
    The median smoothing helps filter hot pixels and 
    cosmic rays. The median is taken by using bin_median
    with a smaller_by_factor=16'''
    image[np.isnan(image)]=np.median(image[~np.isnan(image)])
    binned=bin_median(image,smaller_by_factor=16)
    y,x=np.transpose((binned==binned.max()).nonzero())[0]
    y*=16
    x*=16
    while True:
        x0=max(x-15,0)
        y0=max(y-15,0)
        patch = image[y0:min(y+15,image.shape[0]),x0:min(x+15,image.shape[1])]
        dy,dx=np.transpose(np.where(patch==patch.max()))[0]
        dy-=(y-y0)
        dx-=(x-x0)
        y=min( max(y+dy,0), image.shape[0]-1)
        x=min( max(x+dx,0), image.shape[1]-1)
        if (dx==0) and (dy==0):
            break
    return y,x
   

def align_cut(images,imuse='median',offset=[0,0],cut=True,ncpu=2,keepfrac=0.7):
    '''Align the images and cut them to the minimum overlapping region
    in full pixels, if cut=True.
    in:
    images: 3D np array: iim,y,x
    imuse: iim to align to. Median aligns to median
    cut: decide wether to cut to overlapping region
    offset: [y,x] an offset to add to all images. e.g. to center an object
    ncpu: number of cores to use
    return:
    3D np array with iim,y',x' Where y',x'<y,x dependant on maximal movement 
    or same size as input array if cut ==false.
    '''
    offset = np.array(offset)
    first_median=np.median(images,axis=0)
    if imuse=='median':
        imref = first_median
    else:
        imref = images[imuse,::]
    
    pool=Pool(ncpu)
    get_shifts = subreg(imref)
    first_shifts=pool.map(get_shifts,images)
    pool.close()

    for hh in range(len(images)): 
        images[hh]=shift(images[hh], first_shifts[hh])
        
    #/////////////////////////////////////////////////////////
    #keep only the best of images
    #/////////////////////////////////////////////////////////
    cross_reg=[]
    for ii in range(images.shape[0]):
        im = images[ii,::]
        cross_reg.append(np.sum((im-first_median)**2.))
        
    sorted_cross_reg=np.argsort(cross_reg)
    selected_cross_reg=sorted_cross_reg[0:int(keepfrac*len(images))]
    n_selected=len(selected_cross_reg)

    images = np.array(images)[selected_cross_reg]
    second_median=np.median(images,axis=0)

    print('second subreg')
    pool=Pool(ncpu)
    get_shifts=subreg(second_median)
    second_shifts=pool.map(get_shifts,images)
    pool.close()

    for hh in range(n_selected):
        second_shifts[hh] += offset
        images[hh,::] = shift(images[hh,::],second_shifts[hh])

    if cut:
        total_shifts = []
        for hh in range(n_selected):
            total_shifts.append(np.array(first_shifts)[selected_cross_reg][hh] +\
                           np.array(second_shifts)[hh] +\
                           offset)
        total_shifts = np.array(total_shifts)
        cl = int(np.ceil(np.max(np.hstack((total_shifts[:,1],0))))) #cut left
        cr = int(np.floor(np.min(np.hstack((total_shifts[:,1],0)))))
        cb = int(np.ceil(np.max(np.hstack((total_shifts[:,0],0)))))
        ct = int(np.floor(np.min(np.hstack((total_shifts[:,0],0)))))

        if cr == 0: cr = images.shape[2]+1
        if ct == 0: ct = images.shape[1]+1

        images = images[:,cb:ct,cl:cr]

    return images
