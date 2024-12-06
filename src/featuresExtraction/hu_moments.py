import skimage as sk
import numpy as np

def hu_moments(img):
    '''
    Hu Moments ( or rather Hu moment invariants ) are a set of 7 numbers calculated using central 
    moments that are invariant to image transformations. The first 6 moments have been proved to be 
    invariant to translation,scale, and rotation, and reflection. While the 7th moment’s sign 
    changes for image reflection.

    img: binary image, pixel values 0:1
    return: 7 Hu moments
    '''
    mu = sk.moments_central(img)
    nu = sk.moments_normalized(mu)
    moments = sk.measure.moments_hu(nu)
    return moments