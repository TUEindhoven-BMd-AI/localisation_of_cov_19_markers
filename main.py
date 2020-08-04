import os
import argparse
from datetime import datetime, date
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import callbacks_segmentation
import metrics_segmentation
import matplotlib
matplotlib.use('Agg')
from defaultParams import load_default_params
parser = argparse.ArgumentParser(description='Lung segmentation for COVID-19')

# Dataloading params
#TODO remove defaults
parser.add_argument('--dataPath', default=r"E:\COVID\Covid-datav4",  type=str, help='Indicate the absolute path to the folder than contains the train and validation subfolders')
parser.add_argument('--probe_type', default='convex', type=str, help='Indicate the type of probe as a string. Options are: linear or convex')
parser.add_argument('--batch_size', type=int, help='Indicate the batch size for training')
parser.add_argument('--no_shuffle', default=False, action='store_true', help='Enables shuffling of the training data')
parser.add_argument('--preprocessFunc', default='normalize', type=str, help='Indicate name of the preprocessing function to be used on the images')
parser.add_argument('--augment', type=float, default=0.33, help='Ratio with which any of the augmentation is selected for an image.')
parser.add_argument('--augmentation_strength', type=float, default=1.5, help='Value that indicatese the strength of augmentations. So 1.1 means that the default augmentation settings are increased by 10% and 0.9 means a decrease of strength with 10%.')

# Model params
parser.add_argument('--segmentation_model', default='unet', help='Indicate which segmentation model to train from {unet, unet_plusplus, deeplabv3plus}')
parser.add_argument('--init_weights', default='imagenet', help='Choose weight initialization')
parser.add_argument('--dropout_rate', default='default', help='Set the dropout rate for final layer of the network. If set to default, it will change in the default value corresponding to a model, given in defaultParams.py')
parser.add_argument('--batchnorm', default='default', help='Enables batch normalization if set to true. If set to default, it will change in the default value corresponding to  model, given in defaultParams.py')

# Training params
parser.add_argument('--lr', type=float, help='Indicate the learningrate')
parser.add_argument('--loss', default='categorical_crossentropy', help='Indicate the loss function')
parser.add_argument('--epochs' , type=int, help='Indicate the number of training epochs')
parser.add_argument('--gpu_nr',  type=int, help='Indicate the gpu to be used if gpus available')

# Saving params
parser.add_argument('--savedir', type=str, help='Indicate the absolute path where to save output')
parser.add_argument('--comment', type=str, help='Add additional suffix to the savedir')

args = parser.parse_args()
paramDict = vars(args)

# Create unique version name
today = date.today().strftime("%d/%m/%Y").split('/')
time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")[-8:].split(":")
versionName = os.path.join(paramDict['probe_type']+"_"+today[0]+today[1]+"_"+time[0]+time[1])

# Create a save directory
if paramDict['comment']:
    savedir = Path(args.savedir) / (versionName + '_' + paramDict['comment'])
else:
    savedir = Path(args.savedir) / versionName
paramDict['savedir'] = str(savedir)

"""
DATA LOADING
"""
dataPath = Path(paramDict['dataPath'])/paramDict['probe_type']
paramDict['dataPath'] = str(dataPath)

"""
Set default parameters and settings for the given model and probe_type
"""
load_default_params(paramDict)

# GPU settings
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    from gpuSetting import gpu_usage
    gpu_usage(gpus, int(paramDict['gpu_nr']))

#%%
from Dataloader import training_dataloaders
[train_set, val_set] = training_dataloaders(paramDict, dataPath)

#%%

"""
MODEL DEFINITION
"""
from models import unet, unet_plusplus_resnet50, deeplabv3plus

if paramDict['segmentation_model'] == 'unet':
    model = unet.model_init(paramDict)
elif paramDict['segmentation_model'] == 'unet_plusplus':
    model = unet_plusplus_resnet50.model_init(paramDict)
elif paramDict['segmentation_model'] == 'deeplabv3plus':
    model = deeplabv3plus.model_init(paramDict)
else:
    model = unet.model_init(paramDict) #Else, use standard U-net

#%%
"""
Callback and metrics definition
"""

callbacks = callbacks_segmentation.init_callbacks(val_set, paramDict['num_classes' ], paramDict['train_steps_per_epoch'], paramDict['val_steps_per_epoch'], paramDict['val_freq'], savedir, paramDict)
metrics = metrics_segmentation.init_metrics(paramDict)
#%%
"""
LOSS AND OPTIMIZER DEFINITION
"""
loss = 'categorical_crossentropy'
opt = Adam(lr=paramDict['lr'])
model.compile(optimizer=opt,
              loss=loss,
              metrics=metrics)

"""
TRAINING
"""
print('Start training model {}'.format(versionName))
print('Output can be found in folder: {}'.format(str(savedir)))
print('==========================================================================')

# Create save directory per model
if not os.path.isdir(savedir):
    os.makedirs(savedir)

# Save the parameter dictionary
(savedir / 'parameterSettings.txt').write_text(str(paramDict))
val_set = val_set.batch(paramDict['batch_size'])

# Train the model
history = model.fit(
        train_set,
        steps_per_epoch=paramDict['train_steps_per_epoch'],
        epochs=paramDict['epochs'],
        validation_data=val_set,
        validation_steps = paramDict['val_steps_per_epoch'],
        validation_freq = paramDict['val_freq'],
        workers = 1,
        callbacks = callbacks,
        verbose=1)


"""
SAVING MODEL
"""
tf.keras.models.save_model(model, savedir/"savedWeights"/'FinalWeights.h5', include_optimizer=True)
print('Saved ', versionName, ' in ', savedir)
