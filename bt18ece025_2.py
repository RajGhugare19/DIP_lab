import numpy as np
import cv2 

def conv(img,kernel,stride,padding):
    shape_x = img.shape[0]
    shape_y = img.shape[1]
    kernel_ = np.flipud(np.fliplr(kernel))
    shape_k = kernel.shape
    
    shape_k = kernel.shape
    
    stride_x = stride[1][0]
    stride_y = stride[0][0]
   
    padding_x = padding[1][0]
    padding_y = padding[0][0]
    
    out_x = ((shape_x + 2*padding_x- shape_k[0])//stride_x) + 1
    out_y = ((shape_y + 2*padding_y- shape_k[1])//stride_y) + 1
    
    output = np.zeros([out_x,out_y])

    padded_img = np.zeros([shape_x+2*padding_x,shape_y+2*padding_y])
    padded_img[padding_x:padding_x+shape_x,padding_x:padding_y+shape_y] = img

    for row in range(out_x):
        for col in range(out_y):
            temp =  np.sum(padded_img[row*(stride_x):row*(stride_x)+shape_k[0],col*(stride_y):col*(stride_y)+shape_k[1]]*kernel_)
            output[row,col] = temp
    return output
    
def convolution2d(img,kernel,stride,padding):
    shape = img.shape
    
    if len(shape)==2:
        c = conv_1d(img,kernel,stride,padding)
    else:
        c = []
        for i in range(3):
            c.append(conv(img[:,:,i],kernel,stride,padding))
        c=np.stack(c,axis=2)
    return c

def correlation2d(img,kernel,stride,padding):
    shape = img.shape
    kernel = np.flipud(np.fliplr(kernel))
    if len(shape)==2:
        c = conv_1d(img,kernel,stride,padding)
    else:
        c = []
        for i in range(3):
            c.append(conv(img[:,:,i],kernel,stride,padding))
        c=np.stack(c,axis=2)
    return c

