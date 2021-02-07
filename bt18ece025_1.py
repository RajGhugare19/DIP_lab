import numpy as np

def histo(img):
    hist = np.zeros([2,256],dtype=np.int)
    hist[0] = np.arange(256)
    for i in range(256):
        hist[1,i] = np.sum(np.where(img==i))
    return hist
 
def calculate_metrics(img):
    
    len = img.shape
    sum = img.sum()
    num = np.prod(len)
    mean = sum/num

    sd_hat = (img-mean)**2
    sd = sd_hat.sum()/num

    if len == 2:
        hist = histo(img)

    else:
        hist = np.zeros([2,256,3],dtype=np.int)        
        hist1 = histo(img[:,:,0]) 
        hist2 = histo(img[:,:,1]) 
        hist3 = histo(img[:,:,2]) 
        hist = np.stack((hist1,hist2,hist3),axis=2)
    
    return mean,sd,hist
 
def normalisation(img,pn=True,pc=False,ps=False):
    range=np.max(img)-np.min(img)
    len = img.shape
    sum = img.sum()
    num = np.prod(len)
    mean = sum/num
    sd_hat = (img-mean)**2
    sd = sd_hat.sum()/num
    
    if pn == True:
        norm = (img-np.min(img))/range
    elif pc == True:
        norm = (img-mean)
    elif ps == True:
        norm = (img-mean)/sd
    output_normalised_image = norm
    mean_output_image = norm.sum()/num
    range_output_image = np.array([np.max(norm),np.min(norm)])
    variance_output_image = ((norm-mean_output_image)**2)/num
    return output_normalised_image, mean_output_image, range_output_image,variance_output_image
