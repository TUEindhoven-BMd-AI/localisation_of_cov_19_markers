import numpy as np

def load_default_params(paramDict):
    """
    param paramDict: the dictionary containing all the parameter settings at this moment.
    
    This function updates this parameter dictionary with some defaults
    """    
    # Linear vs convex    
    if paramDict['probe_type'] == 'convex':
        paramDict['input_size'] = (200,260,3)
        
    elif paramDict['probe_type'] == 'linear':
        paramDict['input_size'] = (288,180,3)
        
    else:
        print('ERROR: can not detect linear or convex probe type')
        
 
    if paramDict['batch_size'] is None:
        paramDict['batch_size'] = 32
    if paramDict['lr'] is None:
        paramDict['lr'] = 1e-5
    if paramDict['epochs'] is None:
        paramDict['epochs'] = 200
        
    paramDict['train_steps_per_epoch'] = 100
    paramDict['val_freq'] = 1 #Indicate how often (in terms of epochs) we want to apply validation during training
    
    # Set random gpu if not set manually
    if paramDict['gpu_nr'] is None:
        paramDict['gpu_nr'] = np.random.randint(0,4)
        
        
    if paramDict['dropout_rate'] == 'default':
        if paramDict['segmentation_model'] == 'unet_plusplus':
            paramDict['dropout_rate'] = 0.2
        elif paramDict['segmentation_model'] == 'deeplabv3plus':
            paramDict['dropout_rate'] = 0.1
        else: #standard u-net
            paramDict['dropout_rate'] = 0.5
            
    if paramDict['batchnorm'] == 'default':
        if paramDict['segmentation_model'] == 'unet_plusplus':
            paramDict['batchnorm'] = False
        elif paramDict['segmentation_model'] == 'deeplabv3plus':
            paramDict['batchnorm'] = True
        else: #standard u-net
            paramDict['batchnorm'] = False
            
