#import matplotlib 
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np


from skimage import measure
from skimage.morphology import disk
from scipy.signal import savgol_filter
from scipy.ndimage import morphology


def display_images(xval,yval,preds,savedir, saveVectorIm=False, save=False,  preds_logits= [], uncertainty=[]):
    
    sz = np.shape(preds)
    N_images = np.shape(preds)[0]
    N_classes = np.shape(preds)[3]
    
    if N_classes == 2:
        ColorMap = [[0,0,0],
                    [255,0,0]]
    else:
        ColorMap = [[0,0,0],
                    [0,0,255],
                    [0,255,0],
                    [255,128,0],
                    [255,0,0]]
    
    ColorMap = np.array(ColorMap)
    
    contours = np.zeros((N_images,sz[1],sz[2]))
    # Process data
    for im in range(N_images):
        pred = preds[im,:,:,:]
        
        seg = np.argmax(pred,axis=2)
        mask = ColorMap[seg]
        
        if N_classes == 2:
            unhealty_merged = seg[:,:]>=1
        else:
            unhealty_merged = seg[:,:]>=2
            
            
        unhealty_merged = morphology.binary_closing(unhealty_merged,structure=disk(7))
        
        contour = measure.find_contours(unhealty_merged,0.5)
        
        for cont in contour:
                cont = np.array(cont, np.int32)
                contours[im,cont[:,0],cont[:,1]] = 1
           
            
        if np.array(uncertainty).size:
            plot_masks_and_contours_per_im(xval, yval, im, mask, contour, contours[im], ColorMap, uncertainty[im])        
        else:
            plot_masks_and_contours_per_im(xval, yval, im, mask, contour, contours[im], ColorMap, uncertainty)        


                 
        #Save first image from the val set every epoch       
        
        if save == True:          
            if np.array(uncertainty).size:
                plt.savefig(savedir+'/results_MCdropout_{}.png'.format(im), dpi=100, bbox_inches='tight')    
                if saveVectorIm:
                    plt.savefig(savedir/'results_MCdropout_{}.svg'.format(im), dpi=100, bbox_inches='tight')    
                plt.close()      
            else:
                plt.savefig(savedir+'/results_{}.png'.format(im), dpi=100, bbox_inches='tight')    
                if saveVectorIm:
                    plt.savefig(savedir/'results_{}.svg'.format(im), dpi=100, bbox_inches='tight')    
                plt.close()
        
    
def plot_masks_and_contours_per_im(xval, yval, im, mask, contour, contours, ColorMap, uncertainty):

    if np.array(uncertainty).size:
        W = 18
        n = 5
    else:
        W = 15
        n = 4
        
    plt.figure(figsize=(W, 8))
    
    #Show US image 
    plt.subplot(1,n,1)
    us_image = (np.array(xval[im,:,:,:])/2)+0.5
    plt.imshow(us_image)
    plt.title('B-mode')
    plt.axis('off')
    
    #Show labels
    plt.subplot(1,n,2)
    mask_target = np.argmax(yval[im,:,:,:],axis=2)
    mask_target = ColorMap[mask_target]
    masked_target = 0.7*us_image+0.3*mask_target/255
    plt.imshow(masked_target)
    plt.title('Labels')
    plt.axis('off')
    
    #show US image with contour
    plt.subplot(1,n,3)
    masked_im = 0.7*us_image+0.3*mask/255
    plt.imshow(masked_im)
    
    for con in contour:
        W = int(np.minimum(np.size(con,0)-1,np.floor(0.1*np.size(con,0))))
        if np.mod(W,2)==0:
            W=W+1   
        if W>3:
            con[:,1] = savgol_filter(con[:,1],W,3)
            con[:,0] = savgol_filter(con[:,0],W,3)    
            con = np.concatenate((con,np.expand_dims(con[0,:],0)),0)
            plt.plot(con[:,1],con[:,0],'k')    
    
    
    plt.title('Semantic segmentation')
    plt.axis('off')

    #show US image with labels
    plt.subplot(1,n,4)

    plt.imshow(us_image[:,:,0],cmap='gray')
    plt.axis('off')

    
    for con in contour:
        W = int(np.minimum(np.size(con,0)-1,np.floor(0.1*np.size(con,0))))
        if np.mod(W,2)==0:
            W=W+1   
        if W>3:
            con[:,1] = savgol_filter(con[:,1],W,3)
            con[:,0] = savgol_filter(con[:,0],W,3)    
            con = np.concatenate((con,np.expand_dims(con[0,:],0)),0)
            plt.plot(con[:,1],con[:,0],'r')
    plt.title('COVID-19 markers')


    if np.array(uncertainty).size:
        plt.subplot(1,n,5)

        plt.imshow(uncertainty,cmap='Reds',vmin=0,vmax=0.15)
        plt.axis('off')
        plt.title('Positive score uncertainty')
       
    plt.tight_layout()    

