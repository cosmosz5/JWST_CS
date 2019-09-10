import numpy as np
import pdb
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import scipy.ndimage.filters as ndimage
import cv2

no_data = 5000
input_path = ['/Users/donut/Library/Mobile Documents/com~apple~CloudDocs/JWST_CS/dataset/forms/']
imsize=71

pos1 = np.random.uniform(10, 60, size=int(no_data/3))
pos2 = np.random.uniform(10, 60, size=int(no_data/3))
for i in range(int(no_data/5)):
    im = np.zeros([imsize, imsize])
    im[int(pos1[i]), int(pos2[i])] = 1.0
    im_gauss = ndimage.gaussian_filter(im, sigma=(np.random.random(1)[0]*2))
    im_gauss = im_gauss/np.sum(im_gauss)
    pyfits.writeto(input_path[0]+'gaussians_'+str(i)+'.fits', im_gauss, overwrite=True)


pos1 = np.random.uniform(10, 60, size=int(no_data/2))
pos2 = np.random.uniform(10, 60, size=int(no_data/2))
for i in range(int(no_data/2)):
    im = np.zeros([imsize, imsize])
    im[int(imsize / 2), int(imsize / 2)] = 1.0
    im_gauss = ndimage.gaussian_filter(im, sigma=(np.random.random(1)[0]*50))
    im_gauss = im_gauss/np.sum(im_gauss)
    pyfits.writeto(input_path[0]+'c_gaussians_'+str(i)+'.fits', im_gauss, overwrite=True)


pos1 = np.random.uniform(1, 15, size=int(no_data/2))
pos2 = np.random.uniform(10, 60, size=int(no_data/2))
for i in range(int(no_data/2)):
    im = np.zeros([imsize, imsize])
    im2 = np.zeros([imsize, imsize])
    ax1=int(np.random.random(1)[0]*20+1)
    ax2=int(np.random.random(1)[0]*20+1)
    pa=np.random.random(1)[0]*360
    cv2.ellipse(im, (int(imsize / 2), int(imsize / 2)), (ax1,ax2), \
                pa, 0, 360, (1, 1, 1), -1)
    cv2.ellipse(im2, (int(imsize / 2), int(imsize / 2)), (int(ax1*np.random.random(1)[0]*0.7+ax1*0.1), \
                                                          int(ax2*np.random.random(1)[0]*0.7+ax2*0.1)), \
                pa, 0, 360, (1, 1, 1), -1)
    ells = im2 - im
    ells = ells/np.sum(ells)
    pyfits.writeto(input_path[0]+'rings_'+str(i)+'.fits', ells, overwrite=True)


pos1 = np.random.uniform(1, 15, size=int(no_data/2))
pos2 = np.random.uniform(10, 60, size=int(no_data/2))
for i in range(int(no_data/2)):
    im = np.zeros([imsize, imsize])
    im2 = np.zeros([imsize, imsize])
    ax1=int(np.random.random(1)[0]*20)
    ax2=int(np.random.random(1)[0]*20)
    pa=np.random.random(1)[0]*360
    cv2.ellipse(im, (int(imsize / 2), int(imsize / 2)), (ax1,ax2), \
                pa, 0, 360, (1, 1, 1), -1)
    im = im/np.sum(im)
    pyfits.writeto(input_path[0]+'disk_'+str(i)+'.fits', im, overwrite=True)