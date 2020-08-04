from tensorflow.keras.callbacks import ModelCheckpoint
import os
import matplotlib 
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import tensorflow.keras as keras
import numpy as np
import time
import math

import scipy.io as sio
from scipy import ndimage
from skimage import measure

class training_callback(keras.callbacks.Callback):
    def __init__(self , val_freq, savedir, paramDict):
        self.val_freq = val_freq
        self.savedir = savedir
        # Create all the arrays to hold the metrics data
        self.train_loss = []
        self.val_loss = []
        self.train_mse = []
        self.val_mse = []
        self.train_catacc = []
        self.val_catacc = []
        self.train_jaccard = []
        self.val_jaccard = []
        self.train_dice = []
        self.val_dice = []
        
    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):

        # Validation  is not done every epoch, so this code checks whether validation data is present
        
        # Loss metric
        self.train_loss.append(logs.get('loss'))
        #try:
        if not logs.get('val_loss') is None:
            self.val_loss.append(logs.get('val_loss'))

        # MSE metric
        self.train_mse.append(logs.get('mse'))
        #try:
        if not logs.get('val_mse') is None:
            self.val_mse.append(logs.get('val_mse'))
            
        # Categorical Accuracy metric
        self.train_catacc.append(logs.get('categorical_accuracy'))
        #try:
        if not logs.get('val_mse') is None:
            self.val_catacc.append(logs.get('val_categorical_accuracy'))
         
        # Jaccard metric
        self.train_jaccard.append(logs.get('jaccard'))
        #try:
        if not logs.get('val_jaccard') is None:
            self.val_jaccard.append(logs.get('val_jaccard'))
            #            
        # DICE metric
        self.train_dice.append(logs.get('dice'))
        #try:
        if not logs.get('val_dice') is None:
            self.val_dice.append(logs.get('val_dice'))
        
            
        train_plots = [self.train_loss, self.train_mse, self.train_catacc, self.train_jaccard, self.train_dice]
        val_plots = [self.val_loss, self.val_mse, self.val_catacc, self.val_jaccard, self.val_dice]
        
        ylabels = ['cross_entropy', 'mse', 'cat_acc', 'jaccard', 'dice']
      
        fig = plt.figure(figsize=(12, 10))

        gridx = 2
        gridy = int(math.ceil(len(ylabels)/gridx))
        gs = fig.add_gridspec(gridy, gridx)
        
        k = 0
        for i in range(gridy):
            for j in range(gridx):
                
                if k < len(ylabels):
                    ax = fig.add_subplot(gs[i,j])
                    ax.plot(train_plots[k])
                    ax.plot(val_plots[k])
                        
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel(ylabels[k],fontsize=8)
                    ax.legend(['Train','Val'], loc='upper left')
                    
                    if k > 1:
                        ax.set_ylim([0,1])
                    plt.grid()
                    k += 1
  

        fig.savefig(self.savedir/'TrainingGraph.png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        return

       
class inference_callback(keras.callbacks.Callback):
    def __init__(self, xval=0, yval=0, val_freq=0, savedir=0):
        self.xval = xval
        self.yval = yval
        self.savedir = savedir
        return

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def create_mask_and_contour_plots(self, preds, xval, yval, savedir, saveVectorIm=False):
    
        sz = np.shape(preds)
        N_images = np.shape(preds)[0]
        N_classes = np.shape(preds)[3]
        
        if N_classes == 2:
            ColorMap = [[0,0,0],
                        [255,0,0]]
        else:
            ColorMap = [[0,0,0],
                        [0,255,0],
                        [255,150,0],
                        [255,0,0],
                        [150,0,255]]
        
        ColorMap = np.array(ColorMap)
        
        contours = np.zeros((N_images,sz[1],sz[2]))
        # Process data
        for im in range(N_images):
            pred = preds[im,:,:,:]
            
            seg = np.argmax(pred,axis=2)
            mask = ColorMap[seg]
            
            unhealty_merged = seg[:,:]>=2
                
            contour = measure.find_contours(unhealty_merged,0.5)
            
            for cont in contour:
                    cont = np.array(cont, np.int32)
                    contours[im,cont[:,0],cont[:,1]] = 1
                    
            self.plot_masks_and_contours_per_im(xval, yval, im, mask, contours, ColorMap)        
                                  
            #Save first image from the val set every epoch       
            plt.savefig(savedir/'results_{}.png'.format(im), dpi=100, bbox_inches='tight')    
            if saveVectorIm:
                plt.savefig(savedir/'results_{}.svg'.format(im), dpi=100, bbox_inches='tight')    
            plt.close()
    
    
    def plot_masks_and_contours_per_im(self, xval, yval, im, mask, contours, ColorMap):
        plt.figure(figsize=(16, 8))
        
        #Show US image 
        plt.subplot(1,4,1)
        us_image = (np.array(xval[im,:,:,:])/2)+0.5
        plt.imshow(us_image)
        plt.title('B-mode')
        
        #Show labels
        plt.subplot(1,4,2)
        mask_target = np.argmax(yval[im,:,:,:],axis=2)
        mask_target = ColorMap[mask_target]
        try:
            masked_target = 0.7*us_image+0.3*mask_target/255
        except:
            masked_target = 0.7*us_image[4:-4,2:-2]+0.3*mask_target/255
        plt.imshow(masked_target)
        plt.title('Labels')
        
        #show US image with contour
        plt.subplot(1,4,3)
        try:
            masked_im = 0.7*us_image+0.3*mask/255
        except:
            masked_im = 0.7*us_image[4:-4,2:-2]+0.3*mask/255
        plt.imshow(masked_im)
        plt.title('Segmentation')
        
        #show US image with labels
        plt.subplot(1,4,4)
        contour_image = us_image
        try:
            contour_image[:,:,0] = np.maximum(us_image[:,:,0],contours[im,:,:])
        except:
            contour_image[4:-4,2:-2,0] = np.maximum(us_image[4:-4,2:-2,0], contours[im,:,:])
        plt.imshow(contour_image)
        plt.title('COVID-19 contour')

                       
    def on_epoch_end(self, epoch, logs=None):
        
        
        start = time.time()
        preds = self.model.predict(self.xval)
        end = time.time()
        
        sz = np.shape(preds)
        print("Inference on {} images took {} seconds- {}s per image".format(sz[0], end - start, (end-start)/sz[0]))
        
        self.create_mask_and_contour_plots(preds, self.xval, self.yval, self.savedir)
       

def init_callbacks(val_set, num_classes, steps_per_epoch, val_steps_per_epoch, val_freq, savedir, paramDict):
    
    callbacks = []

    # create checkpoint callback
    if not (savedir/"savedWeights").is_dir():
        os.makedirs(savedir/"savedWeights")
        
    cp_callback = ModelCheckpoint(os.path.join(savedir/"savedWeights",'weights_epoch_{epoch:02d}.hdf5'),
        monitor='val_dice',
        verbose=1, save_best_only=True,
        save_weights_only=False,
        mode='max',
        save_freq='epoch')
    callbacks.append(cp_callback)
    
    if not (savedir/"savedWeights_interval").is_dir():
        os.makedirs(savedir/"savedWeights_interval")
    
    cp_callback_interval = ModelCheckpoint(os.path.join(savedir/"savedWeights_interval",'weights_epoch_{epoch:02d}.hdf5'),
        monitor='val_dice',
        verbose=1, save_best_only=False,
        save_weights_only=False,
        mode='max',
        save_freq='epoch',
        period=5)
    callbacks.append(cp_callback_interval)
    
    
    # create plotting callback
    if not (savedir/"segmentations").is_dir():
        os.makedirs(savedir/"segmentations")
        
    if not (savedir/"segmentations"/"all").is_dir():
        os.makedirs(savedir/"segmentations"/"all")
        
        
    if not (savedir/"segmentations"/"inference").is_dir():
        os.makedirs(savedir/"segmentations"/"inference")
    
    nr_validation_images = 16
    val_set_copy = val_set.shuffle(int(1e6)).batch(nr_validation_images)
    val_iter = val_set_copy.__iter__()
    (xval, yval) = next(val_iter)


    # Callback for plotting loss and metrics
    loss_callback = training_callback(val_freq, savedir, paramDict)
    inference_cb = inference_callback(xval, yval, val_freq, savedir/"segmentations"/"inference")
                                                                                   
    callbacks.append(loss_callback) 
    callbacks.append(inference_cb)
    
    return callbacks
