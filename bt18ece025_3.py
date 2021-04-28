import numpy as np
import cv2 
import matplotlib.pyplot as plt

def gamma_correction(input_image,gamma):
    image = cv2.imread(input_image)/256
    shp = image.shape
       
    if len(shp)==3:
        out = np.zeros(shp)
        for i in range(3):
            out[:,:,i] = image[:,:,i]**(1/gamma)
    elif len(shp)==2:
        out = np.zeros(shp)
        out = image**(1/gamma)
    
    return out*255 
 
 
def contrast_stretching(input_image,max_val,min_val):
    
    image = cv2.imread(input_image)
    shp = image.shape
    
    if len(shp)==3:
        out = np.zeros(shp)
        for i in range(3):
            
            min_i = np.min(image[:,:,i])
            max_i = np.max(image[:,:,i])
            out[:,:,i] = (image[:,:,i]-min_i)*(((max_val-min_val)/(max_i-min_i))+min_val)

    elif len(shp)==2:
            
            min_i = np.min(image)
            max_i = np.max(image)
            out = (image-min_i)*(((max_val-min_val)/(max_i-min_i))+min_val)

    return out/255

def histogram_equilisation(input_image):
    image = cv2.imread(input_image)
    shp = image.shape
    totpix = shp[0]*shp[1]
    if len(shp)==3:
        out = np.zeros(shp)
        for i in range(3):
            pdf = hist(image[:,:,i])/totpix
            cdf_mod = np.floor(np.cumsum(pdf)*(255))
            im_flat = list(image[:,:,i].flatten())
            out_list = [cdf_mod[its] for its in im_flat]
            out[:,:,i] = np.reshape(out_list,(shp[0],shp[1]))            
    elif len(shp)==2:
        pdf = hist(image)/totpix
        cdf_mod = np.floor(np.cumsum(pdf)*(255))
        im_flat = list(image[:,:,i].flatten())
        out_list = [cdf_mod[its] for its in im_flat]
        out = np.reshape(out_list,(shp[0],shp[1]))
    return out


def hist(img):
    img_flat = img.flatten()
    hist = np.zeros([1,256])
    for i in range(256):
        hist[0,i] = np.sum(img_flat==i) 
    return hist


def histogram_stretching(input_image):
    image = cv2.imread(input_image)
    min = np.min(image)
    max = np.max(image)
    out = (((image-min) / (max - min)) * 2**8).astype('uint8')
    return out
