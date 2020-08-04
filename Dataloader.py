import numpy as np
import h5py
import random
import tensorflow as tf
import copy
from pathlib import Path
import scipy.signal as ss
import augmentation.image_augmentation as ia


### Define all the helper functions for the dataloader

  #%%
def init_params(paramDict):
    if paramDict['preprocessFunc'] == 'normalize':
        preprocessFunc = normalize
    else:
        preprocessFunc = lambda x: x
        print('No preprocessing function detected')

    search_key = '_segment' #Added comment to the saved images that contain a segmentation label
    dict_key = 'segmentation_mask' #key used in the save hdf5 file to indicate the segmentation target

    paramDict['num_classes'] = 5 # One class extra, since the background is also seen as a class

    return [preprocessFunc, search_key, dict_key]

def normalize(x):
    # Normalize the images between -1 and 1
    return (2.0*x/255.0) - 1.0

def augmentation(x, y, num_classes, augment_ratio, strength):
    """
    :param x: input data
    :param y: target data
    :param num_classes: number of classes in which each pixel can be classified
    :param augment_ratio: ratio of images (between 0 and 1) that are augmented with each of the given augmentations
    :strength: scaling factor for the default augmentation strength settings
    """
    max_trans = 0.10*strength
    range_for_transforms_params = (1.0-(0.3*strength),1.0+(0.3*strength)) #range between 0.7 and 1.2 for strength=1
    augmentation_funcs = ia.image_augmentation(ratio_augment = 1,
                                  ratio_oneof = augment_ratio,
                                  fliplr=0.5,
                                  flipud=0,
                                  GaussianBlur=(0,.5*strength),
                                  contrast = range_for_transforms_params,
                                  multiply = range_for_transforms_params,
                                  Noise_std=0.01*strength,
                                  translate = (-max_trans,max_trans,-max_trans,max_trans),
                                  rotate=(-10*strength,10*strength),
                                  shear=(-3*strength,3*strength),
                                  scale=range_for_transforms_params
                                  )


    augment = augmentation_funcs.augment_segmentation
    x_aug,y_aug = augment(x,y, num_classes)

    return x_aug, y_aug


class data_loader:

    def __init__(self, dataPath, paramDict):

        # Writing some class variables
        [self.preprocessFunc, self.search_key, self.dict_key] = init_params(paramDict)
        self.paramDict = copy.deepcopy(paramDict)
        self.num_classes = self.paramDict['num_classes']

        # Load the dictionary with all frames and movies split by class
        class_dict = eval((dataPath/(f'class_dictionary{self.search_key}.txt')).read_text())

        # store a copy of the dictionary we are going to use for batch generation. To be able to fill a movie keys again with its frames when it is exhausted
        self.full_class_dict = copy.deepcopy(class_dict)
        self.class_dict = class_dict

        self.classes = [0,0,0,1,2,3]*int(np.ceil((paramDict['batch_size']/6)))


    def process_xy(self, x, y):
        """
        :param x: input data to be preprocessed
        :param y: target data to be preprocessed
        
        Returns preprocessed (augmented and normalized) data
        """
        
        if self.paramDict['augment']:
            x, y = augmentation(x, y, self.paramDict['num_classes'], self.paramDict['augment'], self.paramDict['augmentation_strength'])
            y = np.squeeze(y)

        y_onehot = tf.one_hot(y, self.num_classes, dtype=tf.uint8)
        
        final_x = self.preprocessFunc(x)
        return final_x, y_onehot

    def select_movie(self, rnd_cls):
        """
        :param rnd_cls: one of the classes a pixel can be assigned to
        For a given class rnd_cls, determines the probability of sampling a certain movie based on the amount of available frames and returns a movie_id
        """
        
        # Find the amount of frames per movie in this class
        nr_frames_per_movie = []
        [nr_frames_per_movie.append(len(self.class_dict[rnd_cls][i])) for i in self.class_dict[rnd_cls]]

        # Bound the amount of frames on 50, to not explode probabilities for long movies
        nr_frames_per_movie = np.array(nr_frames_per_movie)
        nr_frames_per_movie_bounded = copy.deepcopy(nr_frames_per_movie)
        nr_frames_per_movie_bounded[nr_frames_per_movie>50] = 50

        # Compute the probabilities
        probs = 0.2*(nr_frames_per_movie_bounded-1)+1

        # Return the sampled movie, sampled using the defined probability distribution
        movie_index = np.argmax(np.random.multinomial(1, probs/np.sum(probs)),-1)
        return movie_index


    def class_balanced_batch_generator(self, batch_size, class_movies):
        """
        :param batch_size: mini-batch size used during training
        :param class_movies:  List containing per class a list of available movie names
        """
        x = []
        y = []

        if not self.paramDict['no_shuffle']:
            random.shuffle(self.classes)

        # Fill the batch with frames from different movies
        # If all movies contributes one frame and we extracted still fewer frames than batch_size, just start again
        for i in range(batch_size):

            # Determine class of next element in batch from precreated list
            rnd_cls = self.classes[i]

            # Select a movie from this class, with a probability that is related to the amount of frames that are still available
            rnd_movie = self.select_movie(rnd_cls)

            #print('set rnd movie: ' , rnd_movie)
            rnd_movie_name = class_movies[rnd_cls][rnd_movie]

            nr_frames = len(self.class_dict[rnd_cls][rnd_movie_name])
            if nr_frames == 0:
                self.class_dict[rnd_cls][rnd_movie_name] = copy.deepcopy(self.full_class_dict[rnd_cls][rnd_movie_name])
                #print(f'Fill again {rnd_movie_name} in class {rnd_cls}')
                nr_frames = len(self.class_dict[rnd_cls][rnd_movie_name])

            rnd_frame = random.randrange(0,nr_frames)
            #print('set rnd frame: ' , rnd_frame)
            frame = self.class_dict[rnd_cls][rnd_movie_name].pop(rnd_frame)

            hf = h5py.File(Path(rnd_movie_name)/(frame+".hdf5"), 'r')

            x.append(np.array(hf['image'], dtype=np.float32))
            y.append(np.array(hf[self.dict_key], dtype=np.float32))

            hf.close()

        # Stack all elements from the batch in an array
        x = np.stack(x)
        y = np.stack(y)
        
        return x,y


    def return_generator(self, batch_size):
        """
        :param batch_size: mini-batch size used during training
        Returns:  A python data generator that yields batches that are balanced either over classes or over movies.
        """

        # List containing per class a list of movie names
        class_movies = []
        for cl in range(4):
            class_movies.append(list(self.class_dict[cl].keys()))

        while True:
            x, y = self.class_balanced_batch_generator(batch_size, class_movies)
            final_x, y_onehot = self.process_xy(x,y)
            yield (final_x,y_onehot)


#%%

def return_full_dataset_loader(dataPath, numpy, paramDict):
    """
    :param dataPath: the absolute path to the dataset to be fully loaded
    :param numpy: Load a numpy array. If false, a tf dataset is loaded
    :param paramDict: the dictionary containing all the parameter settings
    
    Returns a fully loaded dataset that is not split up in batches
    """

    # Function that created np onehot vectors and can deal with nans
    def one_hot(a, num_classes):
        # a is an array of size [bs, x, y] and can contain batch elements of nans
        # Returns [bs, x,y,num_classes], where the nans remain filled with "onehot-nans"
        a[np.isnan(a)] = num_classes #Set the nan entries at class+1, to have this unique code for the one-hot transform
        one_hot_transform = np.concatenate((np.eye(num_classes), np.ones((1,num_classes))*np.nan))
        a_onehot = one_hot_transform[a.flatten().astype(np.uint8)]
        return a_onehot.reshape(a.shape+(num_classes,))

    [preprocessFunc, search_key, dict_key] = init_params(paramDict)

    datasets = []
    dataPaths = [dataPath / 'val', dataPath / 'test']
    for dataPath in dataPaths:

        framesWithLabelInSet = list(dataPath.rglob(f"*{search_key}*.hdf5"))
        target_shape = (len(framesWithLabelInSet),paramDict['input_size'][0],paramDict['input_size'][1])
    
        x = np.zeros((len(framesWithLabelInSet),paramDict['input_size'][0],paramDict['input_size'][1],paramDict['input_size'][-1]), dtype=np.float32)
        y  = np.zeros(target_shape, dtype=np.float32)
    
        for frame_id, frame in enumerate(framesWithLabelInSet):
    
            hf = h5py.File(frame, 'r')
            x[frame_id] = np.array(hf['image'], dtype=np.float32)
            y[frame_id] = np.array(hf[dict_key], dtype=np.float32)
            hf.close()
    
        y = (y+1).astype(np.uint8) #background is now class0, and all classes are +1 to not have negative values
        y = ss.medfilt(np.array(y), [1,7,7]).astype(np.uint8)
    
        # Preprocessing of x
        final_x = preprocessFunc(x)
        y = one_hot(y, paramDict['num_classes'])
        
        test_set = (final_x,y.astype(np.float32))
        if not numpy:
            test_set = tf.data.Dataset.from_tensor_slices(test_set).prefetch(tf.data.experimental.AUTOTUNE)
    
        datasets.append(test_set)

    combined_set_x = np.concatenate((datasets[0][0],datasets[1][0]),axis=0)
    combined_set_y = np.concatenate((datasets[0][1],datasets[1][1]),axis=0)
    return (combined_set_x,combined_set_y)    




def def_target_shape(paramDict):  
    return [None,paramDict['input_size'][0], paramDict['input_size'][1],paramDict['num_classes']]

def training_dataloaders(paramDict, dataPath):
    """
    :param paramDict:   A dictionary containing all the parameters set by the user
    :param dataPath:    Absolute path to the data folder

    Returns
    tf.datasets for training and validation
    """
 
    mydataloader = data_loader(dataPath / 'train', paramDict,)
    train_gen = lambda: mydataloader.return_generator(paramDict['batch_size'])
    train_set = tf.data.Dataset.from_generator(train_gen,
                                               output_types = (tf.float32, tf.float32),
                                               output_shapes = (tf.TensorShape([None,paramDict['input_size'][0],paramDict['input_size'][1],paramDict['input_size'][2]]), tf.TensorShape(def_target_shape(paramDict)))
                                               ).prefetch(tf.data.experimental.AUTOTUNE)


    # Create validation set and set some validation-related parameters in the ParamDict
    val_set = return_full_dataset_loader(dataPath / 'val', paramDict)
    validation_set = tf.data.Dataset.from_tensor_slices(val_set).prefetch(tf.data.experimental.AUTOTUNE)

    paramDict['val_size'] = val_set[0].shape[0]
    paramDict['val_steps_per_epoch'] = int(np.ceil(paramDict['val_size'] / paramDict['batch_size']))

    return [train_set, validation_set]


def return_testset(paramDict, numpy=True):
    """
    :param paramDict: a dictionary containing all the parameter settings
    :param numpy: a boolean indicating whether to return a numpy dataset. If false, a tensorflow dataset is returned
    """
    
    # Returns a np array that contains all data of the test set
    return  return_full_dataset_loader(Path(paramDict['dataPath']), numpy, paramDict)

