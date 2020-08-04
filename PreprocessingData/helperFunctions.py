from pathlib import Path
from random import random
import scipy.io
from PIL import Image, ImageOps
import numpy as np
import h5py

" This file contains helper functions that are used in the mainPreprocessing.py file "

def get_list_of_files(dirName, exts):
    """
    :param exts: list of extensions of files that should be added (incl. the dot)
    
    Returns: a list of files with extension in exts, that are in the shared google drive folder (=dirName)
    """

    allFiles = []
    
    # Iterate over all the entries
    directory = Path(dirName).absolute()

    for entry in directory.iterdir():
        # Only proceed for folders that are not called Unconfirmed or Costole
        if str(entry).find("Unconfirmed") == -1 and str(entry).find("Costole") == -1 :
            
            # If entry is a directory then get the list of files in this directory 
            if entry.is_dir():
                #print('full path: ' , fullPath)
                allFiles = allFiles + get_list_of_files(entry, exts)
            else:
                #print('ext: ' , (fullPath.split(".")[-1]))
                if (entry.suffix in exts) and not ("_score" in entry.stem):
                    #print('ext: ' , (fullPath.split(".")[-1]))
                    allFiles.append(entry)
                    
    return allFiles


def get_patient_path(basePath, filePath):
    """
    :param basePath: Path of the shared google drive folder with the original data
    :param filePath: Absolute path to a mat file for which we want to find the corresponding patient
    
    Returns: Absolute path until the subfolder that is called "Paziente X"
        
    """
    patientPath = filePath.parent

    # Recursively search for the subfolder that starts with Paziente
    while not patientPath.name.startswith("Paziente"):
        patientPath = patientPath.parent
        if patientPath == basePath:
            return None
    return patientPath

def patientPaths_to_filePaths(ProcessedDict):
    """
    :param ProcessedDict: Dictionary with the information regarding data that already has been preprocessed
        
    Returns: A list of full paths to all already processed mat (video) files
    """
    
    allProcessedFiles = []
    
    for patientPath, patient_info in ProcessedDict.items():
        for rel_fileName in patient_info['Movies']:
            allProcessedFiles.append(patientPath/rel_fileName)

    return allProcessedFiles


def determine_dataset(savedir, linearConvex):
    
    numtrain = len(list((savedir / linearConvex /'train').rglob("*.hdf5")))
    numval = len(list((savedir / linearConvex /'val').rglob("*.hdf5")))
    numtest = len(list((savedir / linearConvex /'test').rglob("*.hdf5")))
    
    # Check whether one of the sets contains too little data, otherwise do random split
    if (numval < 0.1*(numtrain+numval+numtest)):
        TrainValTest = 'val'
    elif (numtest < 0.1*(numtrain+numval+numtest)):
        TrainValTest = 'test'
    elif (numtrain < 0.6*(numtrain+numval+numtest)):
        TrainValTest = 'train'
    else:
        rnd = random()
        if rnd <= 0.7:
            TrainValTest = 'train'
        elif 0.7 < rnd <= 0.85:
            TrainValTest = 'val'
        else:
            TrainValTest = 'test'

    return TrainValTest

def load_movie(movie):
    """
    :param movie: Absolute path the the .mat file containing data of the movie
    
    Returns: a numpy array containing the data of the movie
    """
    loadedMatFiles = scipy.io.loadmat(str(movie))
    return loadedMatFiles[list(loadedMatFiles.keys())[-1]]
 
def check_movie_array(movie_mat, linearConvex, patientPath):
    """
    This function checks if the data passes multiple checks regarding its dimensions. If so, we can use this movie in our dataset
    
    :param movie_mat: The array containing the movie data
    
    Returns: boolean for pass or fail
    """

    if len(movie_mat.shape) == 2:
        movie_mat = np.expand_dims(movie_mat,-1)
        movie_mat = np.expand_dims(movie_mat,-1)

    # Either there is only 1 channel, or only 1 frame
    if len(movie_mat.shape) == 3: 
        
        # There are three loaded channels but only 1 frame, two possibilities: x,y,3 (channels or num frames and 1 channel), x,y,frames (1 channel)
        if movie_mat.shape[-1] == 1:
            # 1 channel and 1 frame
            movie_mat = np.expand_dims(movie_mat,-2)
        elif movie_mat.shape[-1] == 3:
            # If last dimension has the value 3 we are unsure wether it is 3 frames or 3 channels,
            # discard unless manually inspected and added to the following list
            patientFilesWith1Frame = [
                    Path(r"C:\Users\s126005\Google Drive\Clinical Study - Study on COVID-19\Gemelli - Roma\Paziente 4"),
                    Path(r"C:\Users\s126005\Google Drive\Clinical Study - Study on COVID-19\Pavia\Paziente 2"),
            ]
            if patientPath in patientFilesWith1Frame:
                # expand frame dimension to 1 frame
                movie_mat = np.expand_dims(movie_mat,-1)
            else:
                print('\n>> Discarded movie: unsure whether 3 frames or 3 channels are available in this movie. Movie dims: {}'.format(movie_mat.shape))
                return (False, movie_mat)
        else:
            # There is only one channel but multiple frames
            movie_mat = np.expand_dims(movie_mat,-2)

    if len(movie_mat.shape) == 4: #Data array has 4 dimensions
        # Only one channel available
        if movie_mat.shape[-2] == 1:
            movie_mat = np.repeat(movie_mat, 3, axis=-2)                
    else:
        print('>> Discarded movie: Strange dimensions detected: {}'.format(movie_mat.shape))
        return (False, movie_mat)

    return (True, movie_mat)


def check_classification_labels(movie_mat, label_mat):
    """
    This function checks if the data and corresponding labels pass multiple checks. If so, we can use the classification labels of this movie in our dataset
    
    :param movie_mat: The array containing the movie data
    :param label_mat: The array containing the classification labels
    
    Returns: categorical labels
    """
    
    passBool = True
    categorical_labels = []
    
    # Some dimension checks
    if len(label_mat.shape) == 2 and label_mat.shape[-1] == movie_mat.shape[-1]:
        categorical_labels = np.cumsum(label_mat,0)[-1]
        
    # There is only 1 frame in the movie, the movie has shape [X,Y,3] and the label file has shape [num_classes,1]
    elif len(label_mat.shape) == 2 and label_mat.shape[-1] == 1 and len(movie_mat.shape) == 3:
        categorical_labels = np.cumsum(label_mat,0)[-1]

    else:
        print('>> Discarded movie: Strange label dimensions detected: {}, with data dims: {}'.format(label_mat.shape, movie_mat.shape))
        passBool = False
        
          
    if passBool:
        # Set categorical labels to -1 which are outside of the valid range between 0 and 3
        categorical_labels[categorical_labels < 0] = -1
        categorical_labels[categorical_labels > 3] = -1
        

    return categorical_labels



def resize_im(origIm, linearConvex, interpolation, background_val):
    """
    :param origIm: np array of frame to be resized
    :param linearConvex: probe type, either convex or linear 
    
    Returns: resized frame dependent on the probe type. 
    Images from the convex array are resized to (wxh) = (260,200), and from the linear array: (wxh) = (180,288)
    """

    SizeLargestAxis = max(origIm.shape[0:2])
    LargestAxisInd = np.argmax(origIm.shape[0:2])
    SizeShortestAxis = min(origIm.shape[0:2])
    ratio = SizeLargestAxis/SizeShortestAxis
    
    # Resize to (W,H) = 260x200 (aspect ratio=1.3)
    if linearConvex == 'convex':
        new_size = (260,200) # in x,y coordinates
        new_ratio = max(new_size)/min(new_size)
        # The image is already wider than higher
        if LargestAxisInd == 1:
            # Fill with extra rows
            if ratio > new_ratio:
               newNrRows = SizeLargestAxis//new_ratio
               delta_h = newNrRows-SizeShortestAxis
               padding = (0, int(delta_h//2), 0, int(delta_h-(delta_h//2)))
               
            # Fill with extra cols
            elif ratio < new_ratio:
                newNrCols = int(SizeShortestAxis*new_ratio)
                delta_w = newNrCols-SizeLargestAxis
                padding = (int(delta_w//2), 0, int(delta_w-(delta_w//2)), 0)
        else:
             newNrCols = int(SizeLargestAxis*new_ratio)
             delta_w = newNrCols-SizeShortestAxis
             padding = (int(delta_w//2), 0, int(delta_w-(delta_w//2)), 0)
             
    # Resize to (W,H) = 180x288 (aspect ratio=1.6)
    else: #linear
        new_size = (180,288) #in x,y coordinates
        new_ratio = max(new_size)/min(new_size)
        if LargestAxisInd == 0:
            # Fill with extra cols
            if ratio > new_ratio:
                newNrCols = SizeLargestAxis//new_ratio
                delta_w = newNrCols-SizeShortestAxis
                padding = (int(delta_w//2), 0, int(delta_w-(delta_w//2)), 0)
            # Fill with extra rows
            elif ratio < new_ratio:
                newNrRows = int(SizeShortestAxis*new_ratio)
                delta_h = newNrRows-SizeLargestAxis
                padding = (0, int(delta_h//2), 0, int(delta_h-(delta_h//2)))
        else:
            newNrRows = int(SizeLargestAxis*new_ratio)
            delta_h = newNrRows-SizeShortestAxis
            padding = (0, int(delta_h//2), 0, int(delta_h-(delta_h//2)))

    # Pad the images to remain the original aspect ratio and resize
    newIm = Image.fromarray(origIm)
    newIm = ImageOps.expand(newIm, padding, fill=background_val)
    
    return newIm.resize(new_size, interpolation)


def save_class_dict(dataFolder, search_key):
    """
    :param dataFolder: Absolute path to the data folder
    :param search_key: The key either _segment or _classif that flags the files that have the corresponding label in our saved preprocessed database.
    
    
    Saves a dictionary in which all saved frames are sorted based on their categorical label. 
    So the dictionary contains different keys for the different categorical labels. Within each key it contains the paths to all frames that correspond to this label.
    This dictionary is used in the dataloader to load class-balanced mini-batches. 
    """
    
    allFiles = list(dataFolder.rglob(f"*{search_key}*.hdf5"))
    
    class_dict = {}
    class_dict[0] = {}
    class_dict[1] = {}
    class_dict[2] = {}
    class_dict[3] = {}
    
    for file in allFiles:
        hf = h5py.File(file, 'r')
        label = int(str(np.array(hf['categorical_label'])))
        hf.close()
        
        if not str(file.parent) in class_dict[label]:
            class_dict[label][str(file.parent)] = []
        class_dict[label][str(file.parent)].append(file.stem)
        
    (dataFolder/(f'class_dictionary{search_key}.txt')).write_text(str(class_dict))


   