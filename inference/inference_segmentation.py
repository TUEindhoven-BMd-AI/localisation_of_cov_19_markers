import os
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
import models
from matplotlib import pyplot as plt
import plot_segmentations
   
tf.compat.v1.disable_eager_execution()

from tensorflow.keras import Model, layers, Input 
def ensembleModels(models, ):
    """
    This function builds an ensemble model from a list of models
    
    Arguments: 
        models: list of Keras models
        
    Return:
        modelEns: ensemble Keras model
        
        
    """    
    model_input = Input(shape=models[0].input_shape[1:])

    # collect outputs of models in a list
    yModels=[model(model_input) for model in models] 
    # averaging outputs
    yAvg=layers.average(yModels) 
    # build model from same input and avg output
    modelEns = Model(inputs=model_input, outputs=yAvg,    name='ensemble')  

    return modelEns 

import tensorflow.keras.backend as K
def predict_with_uncertainty(model, x, no_classes, n_iter=100):
    """
    This function computes uncertainty estimates by Monte Carlo dropout
    
    Arguments:
        model:      Keras model
        x:          model input
        no_classes: number of output classes
        n_iter:     number of MC samples
        
    Return:
        prediction:  model prediction
        uncertainty: model uncertainty
    
    """    
    f = K.function([model.layers[0].input, K.learning_phase()],
               [model.layers[-1].output])
    
    result = np.zeros((n_iter,) + x[:,:,:,0].shape + (no_classes,))
   
    for i in range(n_iter):
        for j in range(np.size(x,0)):
            result[i,j] = f((np.expand_dims(x[j],0), 1))[0]

    prediction = result.mean(axis=0)
    uncertainty = result.std(axis=0)
        
    return prediction, uncertainty

def main(modelName,weightFile,modelFolder,outputFolder, logitboost=[1,1,1,1,1], MCdropout = False, save_images=False, generateVids=False, gpu_nr=0):         
    """
    This file runs inference on one model and saves the inference results, incl. all predictions from the test set in the model folder
    
    arguments:
        modelName:    string containing model foldername or list of strings containing model foldername for ensembling
        weightFile:   string containing weight filename or list of strings containing weight filenames for ensembling
        modelFolder:  string containing model folder dir
        outputFolder: string containing output folder dir
        logitboost:   list containing scalings of the prediction logits. 
        MCdropout:    flag to also perform uncertainty estimation by Monte Carlo Dropout if desired
        save_images:  flag to save images
        generateVids: flag to generate videos
    """

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        from gpuSetting import gpu_usage
        gpu_usage(gpus,gpu_nr)
        
          
    #%%
    # iterate over models
    if isinstance(modelName, list):
        modelnamelist = modelName
    else:
        modelnamelist = [modelName]
        
    if isinstance(weightFile, list):
        weightFileslist = weightFile
    else:
        weightFileslist = [weightFile]        
        
    modellist = []
    for m in range(0,len(modelnamelist)):
        
        loaddir = modelFolder /modelnamelist[m]    
        
        """
        Parameter loading
        """
        paramStr = (loaddir/'parameterSettings.txt').read_text()
        paramDict = eval(paramStr)
                
        """
        Finding newest weight file if not set manually
        """
        weightFile = weightFileslist[m]
        if weightFile is None:
            # Function that finds the latest automatically saved weight file
            def find_newest_weights(path):
        
                allFiles = list(path.rglob("*.hdf5"))
                weightFiles = []
                for file in range(len(allFiles)):
                    if str(allFiles[file].stem)[0:7] == 'weights':
                        weightFiles.append(str(allFiles[file].stem))
        
                epochs = []
                for file in range(len(weightFiles)):
                    epochs.append(int(weightFiles[file].split("_")[-1]))
        
                latestWeightFile = np.argmax(epochs)
                return weightFiles[latestWeightFile]+".hdf5"
        
            weightFile = find_newest_weights(loaddir)
        else:
            ext_weightFile = Path(weightFile).suffix
            if ext_weightFile == '.hdf5' or ext_weightFile =='.h5':
                weightFile = weightFile
            elif not ext_weightFile:
                if weightFile == 'FinalWeights':
                    weightFile = weightFile+".h5"
                else:
                    weightFile = weightFile+".hdf5"
            else:
                print("Indicate a weight file with either .hdf5 or no extension")
                
        
        """
        Model loading
        """    
    
        if paramDict['segmentation_model'] == 'unet':
            model = models.unet.model_init(paramDict)
        elif paramDict['segmentation_model'] == 'unet_plusplus':
            model = models.unet_plusplus_resnet50.model_init(paramDict)
        elif paramDict['segmentation_model'] == 'deeplabv3plus':
            model = models.deeplabv3plus.model_init(paramDict)
            
        model.load_weights(str(loaddir / weightFile))
    
        model._name="Model{}".format(m) 
        modellist.append(model)
    
    
    #%% ensemble the models:
    """
    Model ensembling
    """
    if len(modellist)>1:
        model = ensembleModels(modellist)
    
    model.summary()


    #%%
    """
    Data loading of test set
    """
    paramDict['task'] = 'segmentation'
    settype = 'both'  
    from balancedDataloader import return_testset
    test_set = return_testset(paramDict, numpy=True)

    print('Test set loaded')
        
    #%%
    """
    Inference
    """
    savefigs = save_images
    
    if len(modelnamelist)==1:
        segtype = modelnamelist[0]
    else:
        segtype = 'ensemble_'+'-'.join(modelnamelist)
    
    print('Computing prediction on test set...')
    tarY = np.argmax(test_set[1],-1)
    predY_onehot_ = model.predict(test_set[0], verbose=1)
    
    #% boost outputs for improved sensitivity:
    predY_onehot = np.array(predY_onehot_)    
    
    for i in range(0,np.size(predY_onehot,-1)):
        predY_onehot[:,:,:,i] = predY_onehot[:,:,:,i]*logitboost[i]  

    # ...but ensure that the highest 'score' is never downgraded
    predY = np.argmax(predY_onehot,-1)
    predY_ = np.argmax(predY_onehot_,-1)
    predY[predY_==4] = 4
    predY_onehot = np.array(np.stack([predY==i for i in range(0,np.size(predY_onehot,-1))],-1),dtype='float32')
       
    """ Compute metrics """
    IoverU_cat_list = [np.sum(np.squeeze(np.array(predY==i,dtype=int) * np.array(tarY==i,dtype=int))>0) / np.sum((np.array(predY==i,dtype=int) + np.array(tarY==i,dtype=int))>0) for i in range(0,np.size(test_set[1],-1))]
    IoverU_cat = np.mean(IoverU_cat_list)
    
    Dice_cat_list = [2*np.sum(np.squeeze(np.array(predY==i,dtype=int) * np.array(tarY==i,dtype=int))>0) / (np.sum(np.array(predY==i,dtype=int)) + np.sum(np.array(tarY==i,dtype=int)>0)) for i in range(0,np.size(test_set[1],-1))]
    Dice_cat = np.mean([Dice_cat_list[1],Dice_cat_list[3],Dice_cat_list[4]])
    
    
    IoverU_bin  = np.sum(np.squeeze(np.array(predY>1,dtype=int) * np.array(tarY>1,dtype=int))>0) / np.sum((np.array(predY>1,dtype=int) + np.array(tarY>1,dtype=int))>0)
    Dice_bin  = 2*np.sum(np.squeeze(np.array(predY>1,dtype=int) * np.array(tarY>1,dtype=int))>0) / (np.sum(np.array(predY>1,dtype=int)) + np.sum(np.array(tarY>1,dtype=int)>0))

    acc =  np.sum(predY==tarY)/np.size(predY)
    
    
    print("Mean IoU cat: {}".format(IoverU_cat))
    print("Mean Dice cat: {}".format(Dice_cat))
    print("Mean IoU bin: {}".format(IoverU_bin))
    print("Mean Dice bin: {}".format(Dice_bin))
    print("Mean acc: {}".format(acc))
    
    
    """ Save images and metrics """
    savedir = outputFolder+"/"+segtype+"/"+settype+"/"    
        
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    writeString = f'Test set results:\n\n'
    writeString = writeString + f'Mean IoU cat: {IoverU_cat} \n'
    writeString = writeString + f'Mean IoU bin: {IoverU_bin} \n'
    writeString = writeString + f'Mean Dice cat (across scores 1,3,and 4): {Dice_cat} \n'
    writeString = writeString + f'Mean Dice bin: {Dice_bin} \n'
    writeString = writeString + f'Mean acc: {acc} \n'
    
    text_file = open(savedir+'Test_metrics.txt', "w")
    text_file.write(writeString)
    text_file.close()
    
    
    if savefigs:
        plot_segmentations.display_images(test_set[0],test_set[1],predY_onehot, savedir=savedir,saveVectorIm=False, save=True)
        
            
    #%%
    """
    MC dropout for uncertainty
    """
    if MCdropout: 
        pred_data, uncertainty_data  = predict_with_uncertainty(model, 
                                                                test_set[0], 
                                                                5, 
                                                                n_iter=40)
        uncertainty=np.mean(uncertainty_data[:,:,:,2:],-1)
            
        if savefigs:
            plot_segmentations.display_images(test_set[0],test_set[1],predY_onehot, savedir=savedir,
                                     saveVectorIm=False, save=True,
                                     uncertainty=uncertainty)
            
            plt.xlabel('uncertainty')
            plt.ylabel('Jaccard')

        return acc, Dice_bin, Dice_cat     
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference on trained model for lung segmentation for COVID-19')
    parser.add_argument('--modelName', type=str, help='Indicate the name of the model (folder name)')
    parser.add_argument('--weightFile', default = 'FinalWeights', type=str, help='Indicate the name of the hdf5 weight file, if not indicated the newest automatically saved file will be used')
    parser.add_argument('--modelFolder',   type=str, help='Indicate the folder that contains the models')
    parser.add_argument('--outputFolder',   type=str, help='Indicate the output folder for saving figures and videos')
    parser.add_argument('--save_images', default='false', type=str, help='Indicate if you would like to save all images' )
    parser.add_argument('--MCdropout', default='false', type=str, help='Indicate if you would like to save all images with MCdropout' )
    parser.add_argument('--logitboost', type=str, help='Provide a string containing a list of logit multipliers for inference, e.g. [1,1,1,1.2,1]' )
    parser.add_argument('--gpu_nr', default='0', type=str, help='Indicate which gpu you would like to use')

    args = parser.parse_args()   
    
    modelName = args.modelName
    weightFile = args.weightFile
    modelFolder = args.modelFolder
    outputFolder = args.outputFolder
    
    if args.MCdropout == 'true':
        MCdropout = True
    else:
        MCdropout = False   
        
    if args.save_images == 'true':
        save_images = True   
    else:
        save_images = False
    
    gpu_nr = int(args.gpu_nr)
    
    logitboost = map(float, (args.logitboost).strip('[]').split(','))

    main(modelName=args.modelName,weightFile=args.weightFile,modelFolder=args.modelFolder,outputFolder=args.outputFolder, logitboost=logitboost, MCdropout = MCdropout, save_images=save_images, gpu_nr=gpu_nr)

    
    
    
