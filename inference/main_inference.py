
import inference_segmentation
import numpy as np
from tabulate import tabulate
from pathlib import Path

modelFolder = Path.cwd().parent/'SavedModels'   # Indicate the absolute path to the folder containing the model subfolders
outputFolder = "..."  # Indicate the absolute path to the folder to save the inference output

MCdropout = False   # Whether or not to apply MC dropout for uncertainty estimates
save_images = False # Whether or not to save the images


# comparison of models:
modelList = ["(1)_Unet",
             "(2)_Unet_augmentation",
             "(3)_Unet_pp_augmentation",
             "(4)_Deeplabv3_augmentation"]
weightList = ['FinalWeights','FinalWeights','FinalWeights','FinalWeights']            
logitboost = [[1,1,5,3.5,1],[1,1,5,3.5,1],[1,1,1,1,1],[1,1,8,8,1]]       

# ensemble model:
modelList = ["(2)_Unet_augmentation",
             "(3)_Unet_pp_augmentation",
             "(4)_Deeplabv3_augmentation"]
weightList = [['FinalWeights','FinalWeights','FinalWeights']]
logitboost = [[1,1,2,2,2]]       

acc = np.zeros(len(modelList))
Dice_bin = np.zeros(len(modelList))
Dice_cat = np.zeros(len(modelList))

rows = []
for i in range(0,len(modelList)):
    acc[i], Dice_bin[i], Dice_cat[i] = inference_segmentation.main(modelFolder=modelFolder,
                                                                   modelName=modelList[i],
                                                                   weightFile=weightList[i],
                                                                   MCdropout=MCdropout,
                                                                   logitboost=logitboost[i],
                                                                   save_images=save_images,
                                                                   outputFolder=outputFolder,
                                                                   gpu_nr=0)                          

    rows.append([modelList[i],acc[i], Dice_bin[i], Dice_cat[i]])

print(tabulate(rows, headers=['Name', 'cat Acc', 'Dice', 'cat Dice']))





