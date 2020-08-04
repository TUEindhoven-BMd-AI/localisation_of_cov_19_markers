import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,  BatchNormalization, SeparableConv2D, Conv2D, Conv2DTranspose, Dropout, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import LeakyReLU, concatenate, Activation, Reshape, Add
from tensorflow.keras import regularizers
import tensorflow as tf



def model_init(paramDict):
    
    #Define kernel and pool sizes
    pools = [[2,2],
            [2,2],
            [5,5]]
    
    filterdims =   [[16,3],
                    [32,5],
                    [64,5]]

    input_size =  (paramDict['input_size'][0],paramDict['input_size'][1],paramDict['input_size'][2])  
    output_size = (paramDict['input_size'][0],paramDict['input_size'][1],paramDict['num_classes'])

    model = u_net(input_size, output_size, pool=pools, filterdims=filterdims, out_activation='softmax', batch_norm=paramDict['batchnorm'], dropout=paramDict['dropout_rate'])

    return model   
    

def standardization(x):
    x_norm = tf.map_fn(tf.image.per_image_standardization, x)
    x_norm = tf.stack(x_norm)
    return x_norm

def standardization_output_shape(input_shape):
    return tuple(input_shape)

def u_net(input_dim, target_dim, pool, strides=1, skips=True, skip_type = 'concat', batch_norm=False, dropout=0.5, dropout_allLayers = False, layer_activation = 'relu', out_activation='softmax', filterdims = [], filterdims_out = 3, latentdims=[64,3], prefilterdims = [], separable_convs = False, l2_pen=0, per_image_standardization = False):
    """U-net model

    Args:
        input_dim :      dimensions of the input data (x,x)
        target_dim:      dimensions of the target data (x,x)
        strides:         stride of the first convolution per conv block in encoder; 
                         & stride of the last convolution per conv block in decoder;
        pool:            maxpool dimension
        skips:           True or false; use skip connections or not
        skip_type:       'sum' or 'concat'
        batch_norm:      True or False; apply batch normalization
        dropout:         dropout on the fully connected layers
        dropout_allLayers: use dropout for each layer block or only the latent dimension
        layer_activation: type of activation between conv and batch_norm (e.g. 'relu', 'LeakyReLU')
        out_activation:  type of activation at the output (e.g. 'sigmoid', or None)
        filterdims:      filter dimensionality matrix 
                         (e.g. for 3 layers: [[Nfeat1,dim1],[Nfeat2,dim2],[Nfeat3,dim3]])
        filterdims_out:  kernel dimensions of the filter at the output layer
        latent_dims:     latent filter dimensions
        prefilterdims:   optional set of filters before the encoder
        separable_convs: use depthwise separable convolutions rather than regular convs [Xception: Deep Learning with Depthwise Separable Convolutions] 
        l2_pen:          l2 penalty on the convolutional weights
        per_image_standardization: normalize input images to zero mean unity variance
        
    Returns:
        keras model
    """
    
    inputs = Input(shape=input_dim)

    input_layer = inputs

    
    # make input dimensions compatible with the network; i.e. add a channel dim if neccesary   
    if len(np.shape(input_layer))<4:  
        input_layer = Lambda(lambda x: K.expand_dims(x))(input_layer)  

    if per_image_standardization == True:
        input_layer =  Lambda(standardization,output_shape=standardization_output_shape)(input_layer)

    
    if filterdims == []:
        filterdims =   [[16,3],
                        [32,5],
                        [64,5]]
          

    if len(target_dim)<3:  
        n_classes = 1
    else:
        n_classes = target_dim[-1]  

    last_layer = input_layer
    """
    =============================================================================
        PREFILTER LAYER
    =============================================================================
    """
    for i in range(0,np.size(prefilterdims,0)):
        if separable_convs:
            conv_pre = SeparableConv2D(prefilterdims[i][0], prefilterdims[i][1], activation = None, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(l2_pen),name='preconv{}'.format(i))                
        else:
            conv_pre = Conv2D(prefilterdims[i][0], prefilterdims[i][1], activation = None, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(l2_pen),name='preconv{}'.format(i))        

        conv = last_layer
        
        if batch_norm==True:
            conv = BatchNormalization()(conv)
            
        conv = conv_pre(conv)

        print("conv shape :",conv.shape)

        if layer_activation=='LeakyReLU':
            conv = LeakyReLU(alpha=0.1)(conv)
        else:
            conv = Activation(layer_activation)(conv) 

        if dropout_allLayers and dropout>0:
            conv = Dropout(dropout)(conv)    
            print("dropout layer")
            

        last_layer = conv 
    
    """
    =============================================================================
        ENCODER
    =============================================================================
    """
    convs = []
    convs_a = []
    convs_b = []
    pools= []
    for i in range(0,np.size(filterdims,0)):
        if separable_convs:
            conv_a = SeparableConv2D(filterdims[i][0], filterdims[i][1], strides = strides, activation = None, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(l2_pen), name='conv{}a'.format(i))
            conv_b = SeparableConv2D(filterdims[i][0], filterdims[i][1], activation = None, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(l2_pen),name='conv{}b'.format(i))                
        else:
            conv_a = Conv2D(filterdims[i][0], filterdims[i][1], strides = strides, activation = None, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(l2_pen), name='conv{}a'.format(i))
            conv_b = Conv2D(filterdims[i][0], filterdims[i][1], activation = None, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(l2_pen),name='conv{}b'.format(i))        
        

        conv = last_layer
        if batch_norm==True:
            conv = BatchNormalization()(conv)
            
        conv = conv_a(conv)

        print("conv shape :",conv.shape)

        if layer_activation=='LeakyReLU':
            conv = LeakyReLU(alpha=0.1)(conv)
        else:
            conv = Activation(layer_activation)(conv) 
 
        if batch_norm==True:
            conv = BatchNormalization()(conv)
           
        conv = conv_b(conv)
        print("conv shape :",conv.shape)
    
            
        if layer_activation=='LeakyReLU':
            conv = LeakyReLU(alpha=0.1)(conv)
        else:
            conv = Activation(layer_activation)(conv) 
        
        convs.append(conv)
        pools.append(MaxPooling2D(pool_size=pool[i],name='maxpool{}'.format(i))(conv))
        print("pool shape :",pools[i].shape)

        last_layer = pools[-1]
        
        if dropout_allLayers and dropout>0:
            last_layer = Dropout(dropout)(last_layer)    
            print("dropout layer")
        
   
    """
    =============================================================================
        LATENT LAYER
    =============================================================================
    """   
    if len(latentdims)==2:
        if separable_convs:
            conv_latent = SeparableConv2D(latentdims[0], latentdims[1], activation = None, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(l2_pen), name='Conv1latent_space')(last_layer)
        else:
            conv_latent = Conv2D(latentdims[0], latentdims[1], activation = None, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(l2_pen), name='Conv1latent_space')(last_layer)

            print("conv shape :",conv_latent.shape)

        if layer_activation=='LeakyReLU':
            conv_latent = LeakyReLU(alpha=0.1)(conv_latent)
        else:
            conv_latent = Activation(layer_activation)(conv_latent) 

        if  dropout>0:
            conv_latent = Dropout(dropout)(conv_latent)    
            print("dropout layer")


        if separable_convs:     
            conv_latent = SeparableConv2D(latentdims[0], latentdims[1], activation = None, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(l2_pen), name='Conv2latent_space')(conv_latent)
        else:
            conv_latent = Conv2D(latentdims[0], latentdims[1], activation = None, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(l2_pen), name='Conv2latent_space')(conv_latent)
            
        print("conv shape :",conv_latent.shape)
        
        if layer_activation=='LeakyReLU':
            conv_latent = LeakyReLU(alpha=0.1)(conv_latent)
        else:
            conv_latent = Activation(layer_activation)(conv_latent) 
            
    else:
        conv_latent = last_layer
        print("skipping latent layer..")
        
        if  dropout>0:
            conv_latent = Dropout(dropout)(conv_latent)    
            print("dropout layer")
    
    last_layer = conv_latent
            
    
    """
    =============================================================================
        DECODER
    =============================================================================
    """
    filterdims = filterdims[::-1]
    for i in range(0,np.size(filterdims,0)):
        # 'learned' upsampling (nearest neighbor interpolation with 2x2, followed by conv of 2x2 )
#        up = Conv2DTranspose(filterdims[i][0], 2, activation = None, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(last_layer))
        up = UpSampling2D(name='upsample{}'.format(i),size = pool[-i-1])(last_layer)
        print("up shape   :",up.shape)  
        
        if skips==True:
            # Skip connections
            if skip_type == 'concat':
                merged = concatenate([convs[-1-i],up],3)                
            else:
                merged = Add(name='Skip-connection{}'.format(i))([convs[-1-i], up])   
        else:
            merged = up
            
        print("merge shape:",merged.shape)       


        if batch_norm==True:
            conv = BatchNormalization()(conv)   


        shape_in= merged.shape.as_list()
        

        layer = Conv2DTranspose(filterdims[i][0], filterdims[i][1], activation = None, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(l2_pen), name='deconv{}a'.format(i))
        conv = layer(merged)
         
        shape_out = layer.compute_output_shape(shape_in)     
        conv.set_shape(shape_out)
   
        print("conv shape :",conv.shape)
                
        if layer_activation=='LeakyReLU':
            conv = LeakyReLU(alpha=0.1)(conv)
        else:
            conv = Activation(layer_activation)(conv)   
                        
        if batch_norm==True:
            conv = BatchNormalization()(conv)
        
        if i<np.size(filterdims,0)-1:
            shape_in= merged.shape.as_list()


            layer =  Conv2DTranspose(filterdims[i+1][0], filterdims[i+1][1], strides = strides, activation = None, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(l2_pen), name='deconv{}b'.format(i))
            conv = layer(conv)

            shape_out = layer.compute_output_shape(shape_in)     
            conv.set_shape(shape_out)
    
            if layer_activation=='LeakyReLU':
                conv = LeakyReLU(alpha=0.1)(conv)
            else:
                conv = Activation(layer_activation)(conv)   
    
            print("conv shape :",conv.shape)            
            last_layer = conv       
            
            if dropout_allLayers and dropout>0:
                last_layer = Dropout(dropout)(last_layer)    
                print("dropout layer")
            
        else: # last layer:
            conv = Conv2DTranspose(filterdims[i][0], filterdims[i][1], activation = None, strides = strides, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(l2_pen), name='deconv{}b'.format(i))(conv)             
            if layer_activation=='LeakyReLU':
                conv = LeakyReLU(alpha=0.1)(conv)
            else:
                conv = Activation(layer_activation)(conv) 
                
            conv = Conv2DTranspose(filterdims[i][0], filterdims[i][1], activation = None, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(l2_pen), name='deconv{}c'.format(i))(conv)             
            if layer_activation=='LeakyReLU':
                conv = LeakyReLU(alpha=0.1)(conv)
            else:
                conv = Activation(layer_activation)(conv)  


                
            final_layer = Conv2D(n_classes, filterdims_out, activation = out_activation, padding = 'same', kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(l2_pen), name='Final_conv')(conv) 
            print("conv shape :",conv.shape)  
            
            final_layer = Reshape(target_dim)(final_layer)
        
                    
    model = Model(inputs = inputs, outputs = final_layer)

    return model
