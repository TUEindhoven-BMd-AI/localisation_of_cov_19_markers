"""
This file reads the data from the raw datafolderand saves it in the correct format for the data loader.
It can be run multiple times and checks if new data arrived in the raw datafolder. It will then only process the new data.
"""
import argparse
from pathlib import Path, WindowsPath
from random import random
import scipy.io
from datetime import datetime, date
import scipy.misc
import numpy as np
from PIL import Image, ImageOps
import helperFunctions
import h5py

#%%
parser = argparse.ArgumentParser(description='Loading and saving files from the synced drive folder')

parser.add_argument('--loaddir', typ=str, help='Indicate the absolute path to the folder with the original data')
parser.add_argument('--savedir', type=str, help='Indicate the direct path where to save the processed data')
parser.add_argument('--ProcessedFilesName', default=r"ProcessedFiles.txt", help='Indicate the name of the file in which to save the information regarding which files have been preprocessed')

args = parser.parse_args()

#%%
# Create all folders and subfolders to save the preprocessed data 
basePath = Path(args.loaddir)
savedir  = Path(args.savedir)
processedFilePath = savedir/args.ProcessedFilesName

#%%
# Create output directories
for subfol in ['train', 'val', 'test']:
    (savedir / 'linear' / subfol).mkdir(parents=True,exist_ok=True)
    (savedir / 'convex' / subfol).mkdir(parents=True,exist_ok=True)
    
#%%
# Load or create lookup table (=ProcessedDict) that contains all information regarding allready preprocessed files
if not (processedFilePath).is_file():
    processedFilePath.write_text('{}')
# Per patient key, it contains a TrainValTest key, and a Movies key. This Movies key contains a list of .mat movie-files that have been saved as hdf5 already.
ProcessedDict = eval(processedFilePath.read_text()) 
#%%
"""
Processing of the images
"""
# List of absolute paths to all .mat movie files in the data folder
allMovies = helperFunctions.get_list_of_files(basePath, ['.mat'])

# List of absolute paths to all .mat movie files in the data folder that are already in the ProcessedDict file
ProcessedFiles = helperFunctions.patientPaths_to_filePaths(ProcessedDict)

for movie in allMovies:
    # If this mat file is already processed check if either classification or segmentation labels are still undone
    if movie in ProcessedFiles: 
        continue
    
    # This movie is new 
    else:
        print(f"\n\n Found new movie: {movie}")
        
        # Retrieve the absolute path until the subfolder that starts with "Paziente"
        patientPath = helperFunctions.get_patient_path(basePath, movie)

        # Do some checks before processing this movie and saving it
        if patientPath is None:
            print('\n>> Discarded file: Cannot detect Paziente subfolder')
            continue
    
        linearConvex = movie.stem.split("_")[0]
        if not linearConvex in ['linear', 'convex']:
            print('\n>> Discarded file: cannot detect convex or linear probe')
            continue
        
        # Determine if it is the first movie of this patient. if no, find the earlier determined dataset this patient belongs to
        if patientPath in list(ProcessedDict.keys()):
            TrainValTest = ProcessedDict[patientPath]['TrainValTest']
        else:
            ProcessedDict[patientPath] = {}
            TrainValTest = helperFunctions.determine_dataset(savedir, linearConvex)
            ProcessedDict[patientPath]['TrainValTest'] = TrainValTest
            if 'Movies' not in list(ProcessedDict[patientPath].keys()):
                ProcessedDict[patientPath]['Movies'] = []

        
        # Load movie as array
        movie_mat = helperFunctions.load_movie(movie)
        passMovie, movie_mat = helperFunctions.check_movie_array(movie_mat, linearConvex, patientPath)

        if not passMovie:
            continue
        
        # Create a unique folder name per movie
        if movie.parent == patientPath:
            folderName = savedir/linearConvex/TrainValTest/(patientPath.parent.stem+"_"+patientPath.stem+"_"+movie.stem)
        else:
            folderName = savedir/linearConvex/TrainValTest/(patientPath.parent.stem+"_"+patientPath.stem+"_"+movie.parent.stem+"_"+movie.stem)
            
        folderName.mkdir(parents=True,exist_ok=True)
        
        for frame in range(movie_mat.shape[-1]):
            resized_frame = helperFunctions.resize_im(movie_mat[...,frame], linearConvex, Image.BICUBIC, 0)
            hf = h5py.File(folderName/f'frame_{frame}.hdf5', 'w')
            hf.create_dataset('image', data=resized_frame)
            hf.close()
        
            print(f'\n Saved frame {frame} of movie {movie} in {folderName}')

   
        ProcessedDict[patientPath]['Movies'].append(movie.relative_to(patientPath))
        
        _ = processedFilePath.write_text(str(ProcessedDict))
        print('\nUpdated the processedFiles.txt file')
        
        ProcessedFiles.append(movie)

#%%        

"""
Processing of the classification labels
"""
ProcessedDict = eval(processedFilePath.read_text()) 
ProcessedFiles = helperFunctions.patientPaths_to_filePaths(ProcessedDict)

print("\nSTART CLASSIFICATIONS")
print("===============================================================")
# Loop over all movies that are already saved in the data folder and check if classification labels are already preprocessed, if not, check if labels are available and save them in the correct format
for movie in ProcessedFiles:
    patientPath = helperFunctions.get_patient_path(basePath, movie)
    TrainValTest = ProcessedDict[patientPath]['TrainValTest']
    linearConvex = movie.stem.split("_")[0]

    # Find unique folder name per movie
    if movie.parent == patientPath:
        folderName = savedir/linearConvex/TrainValTest/(patientPath.parent.stem+"_"+patientPath.stem+"_"+movie.stem)
    else:
        folderName = savedir/linearConvex/TrainValTest/(patientPath.parent.stem+"_"+patientPath.stem+"_"+movie.parent.stem+"_"+movie.stem)
        
    allFrames = set(folderName.rglob("*.hdf5"))
    if len(allFrames) == 0:
        print(f">> ERROR {movie}: detected zero saved hdf5 files, while this movie was already in the processed dict")
        continue
    
    frames_with_classification = set(folderName.rglob("*_classif*.hdf5"))
     
    # Loop over all saved frames of this movie and check if they need a classification label
    frames_without_label = list(allFrames - frames_with_classification)
    
    # Returns binary and categorical labels in case the label file did pass the tests
    if len(frames_without_label) > 0:
        movie_mat = helperFunctions.load_movie(movie)
        try:
            label_mat = scipy.io.loadmat(movie.parents[0]/ (movie.stem+"_score.mat"))
            label_mat = label_mat[list(label_mat.keys())[-1]]
        except:
            print(f'>> Skip {movie}: no classification label file could be found for movie {movie}')
            continue
    
        categorical_labels = helperFunctions.check_classification_labels(movie_mat, label_mat)
         
        for frame in frames_without_label:
            # check if classificaiton label is available 
            
            frameNr = int(frame.stem.split("_")[1])
            cat_label = categorical_labels[frameNr].astype(np.uint8)
            
            # Check for invalid labels
            if cat_label < 0:
                print(f'>> Discarded classification label of frame {frameNr} from {movie}, because it changes values outside the valid range')
                continue
            
            else:
                try:
                    hf = h5py.File(folderName/f'frame_{frameNr}.hdf5', 'r+')
                    prevName = folderName/f'frame_{frameNr}.hdf5'
                except:
                    hf = h5py.File(folderName/f'frame_{frameNr}_segment.hdf5', 'r+')
                    prevName = folderName/f'frame_{frameNr}_segment.hdf5'
                hf.create_dataset('categorical_label', data=cat_label)
                hf.close()
                
                # Update saved file name        
                newName = prevName.parent / (prevName.stem+"_classif.hdf5")
                prevName.rename(newName)
                
        
                if newName != prevName:
                    print(f'{movie}: Added classification label for frame {frameNr} in {folderName}')
    else:
        continue
      #%%  

"""
Processing of the segmentation masks
"""
from jsonToMask import convert_json_to_mask


print("\nSTART SEGMENTATIONS")
print("===============================================================")
ProcessedDict = eval(processedFilePath.read_text()) 
ProcessedFiles = helperFunctions.patientPaths_to_filePaths(ProcessedDict)

for movie in ProcessedFiles:
    patientPath = helperFunctions.get_patient_path(basePath, movie)
    TrainValTest = ProcessedDict[patientPath]['TrainValTest']
    linearConvex = movie.stem.split("_")[0]


    # Find unique folder name per movie
    if movie.parent == patientPath:
        folderName = savedir/linearConvex/TrainValTest/(patientPath.parent.stem+"_"+patientPath.stem+"_"+movie.stem)
    else:
        folderName = savedir/linearConvex/TrainValTest/(patientPath.parent.stem+"_"+patientPath.stem+"_"+movie.parent.stem+"_"+movie.stem)
        
    allFrames = set(folderName.rglob("*.hdf5"))
    if len(allFrames) == 0:
        print(f">> ERROR {movie}: detected zero saved hdf5 files, while this movie was already in the processed dict")
        continue
    
    frames_with_segmentation = set(folderName.rglob("*_segment*.hdf5"))    

    movie_mat = helperFunctions.load_movie(movie)

    # Check availability of segmentation masks for this movie
    jsonFolder = movie.with_suffix('.images')    
    availableSegmentationMasks = list(jsonFolder.rglob("*.json"))
    if not availableSegmentationMasks:
        print(f'>> Skip segmentation of {movie}: no segmentation folder could be found')
        continue
    
    # Check for all frames that do not have segmentation labels yet, if such a label is available now
    try:
        annotatedFrames = [int(i.stem.split("_")[-1]) for i in availableSegmentationMasks]
    except:
        print(f">> Discard {movie}: File names of json files for {patientPath} are not according to the standard")
        continue
    for json_file in list(allFrames - frames_with_segmentation):
        frameNr = int(str(json_file.stem).split("_")[1])     
       
        if frameNr in annotatedFrames:
            listIndex = annotatedFrames.index(frameNr)
            # Find the mask as a np array
            mask = convert_json_to_mask(availableSegmentationMasks[annotatedFrames.index(frameNr)])
            resized_mask = helperFunctions.resize_im(mask, linearConvex, Image.NEAREST, -1)

            # Check if the mask contains invalid values
            if np.max(resized_mask) > 3 or np.min(resized_mask) < -1:
                print(f'>> Discarded segmentation mask of frame {frameNr} from file: {movie} because it contains values outside of the valid range')
                continue
                    
            try:
                hf = h5py.File(folderName/f'frame_{frameNr}.hdf5', 'r+')
                prevName = folderName/f'frame_{frameNr}.hdf5'
            except:
                hf = h5py.File(folderName/f'frame_{frameNr}_classif.hdf5', 'r+')
                prevName = folderName/f'frame_{frameNr}_classif.hdf5'
            hf.create_dataset('segmentation_mask', data=resized_mask)
            
            hf.close()
        
            # Update saved file name        
            try:
                newName = prevName.parent / (prevName.stem+"_segment.hdf5")
                prevName.rename(newName)
            except:
                print(f'ERROR: permission error for changing name: {prevName} to {newName}')
        
            if newName != prevName:
                print(f'{movie}: Added segmentation label for frame {frameNr} in {folderName}')
        
            
        
#%% Recreate the new class dictionary 
                
train_folder = savedir /'linear'/'train'  
helperFunctions.save_class_dict(train_folder, '_classif')
helperFunctions.save_class_dict(train_folder, '_segment')

train_folder = savedir /'convex'/'train'  
helperFunctions.save_class_dict(train_folder, '_classif')
helperFunctions.save_class_dict(train_folder, '_segment')

print('Updated the class dictionaries in the train folder')
    
    
