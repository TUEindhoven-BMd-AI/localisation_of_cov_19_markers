import numpy as np
import augmentation.imgaug_utils.augmenters as ia_augmenters
import augmentation.imgaug_utils.imgaug as ia_imgaug

class image_augmentation:
    def __init__(self, ratio_augment = 1, ratio_oneof = 0.5, crop=(0,10), fliplr=0.5, flipud=0.5, GaussianBlur=(0,1), contrast = (0.9,1.1), multiply = (1.1,1.5), Noise_std=0.01, translate = (-10,10), rotate=3, shear=3, scale=1.1):
     
        self.aug_seq = ia_augmenters.Sequential([
            ia_augmenters.Sometimes(ratio_oneof, ia_augmenters.Fliplr(fliplr, name="Flipper")),
            ia_augmenters.Sometimes(ratio_oneof, ia_augmenters.Flipud(flipud, name="Flipperud")),
            ia_augmenters.Sometimes(ratio_oneof, ia_augmenters.Affine(translate_percent={"x": translate[0:2], "y": translate[2:4]}, rotate=rotate, shear=shear, scale=scale, name="Affine")), # angles of rotate and shear in degrees
            ia_augmenters.Sometimes(ratio_oneof, ia_augmenters.ContrastNormalization(alpha=(contrast[0],contrast[1]), name="ContrastNorm")),
            ia_augmenters.Sometimes(ratio_oneof, ia_augmenters.Multiply((multiply[0],multiply[1]),name='multiply')),                
            ia_augmenters.Sometimes(ratio_oneof, ia_augmenters.GaussianBlur(GaussianBlur, name="GaussianBlur")),
            ia_augmenters.Sometimes(ratio_oneof, ia_augmenters.AdditiveGaussianNoise(scale=Noise_std, name="GaussianNoise")),
        ]) 
        self.hooks = ia_imgaug.HooksImages # hooks to turn off several transforms for targets
        self.ratio_augment = ratio_augment
                

    def augment_segmentation(self, inputs, targets, num_classes):

        # deterministic augmentation to apply same transforms on both inputs and targets
        aug_seq_det = self.aug_seq.to_deterministic()

        # change the activated augmenters for binary masks, we only want to execute horizontal crop, flip and affine transformation
        # on the segmentation targets
        hooks_binmasks = self.hooks(activator=self.activator_binmasks)    
        
        nr_augment = int(np.ceil(self.ratio_augment*np.shape(inputs)[0]))
        inds_augment = np.random.permutation(np.arange(0,np.shape(inputs)[0]))[0:nr_augment]
        
        # Apply augmentation on the input
        inputs[inds_augment] = aug_seq_det.augment_images(inputs[inds_augment])  

        # Apply augmentation on the target
        targets_flat = np.expand_dims(targets,-1)
        targets_flat[inds_augment] = aug_seq_det.augment_images(targets_flat[inds_augment],hooks=hooks_binmasks)  
        
        T = 0.05+np.arange(0,num_classes)
        targets = np.digitize(targets_flat, T)
                
        return inputs, targets

        
    def activator_binmasks(self, images, augmenter, parents, default):
        if augmenter.name in ["GaussianBlur", "GaussianNoise", "ContrastNorm", "multiply"]:
            return False
        else:
            # default value for all other augmenters
            return default      

