import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dropout, Conv2D, Conv2DTranspose, LeakyReLU, MaxPooling2D, concatenate, Add, Lambda, Softmax, ZeroPadding2D, BatchNormalization


def convolution_block(x, filters, size, batchnorm, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    if batchnorm:
        x = BatchNormalization()(x)
    if activation:
        x = LeakyReLU(alpha=0.1)(x)
    return x

def residual_block(blockInput, batchnorm, num_filters=16):
    x = LeakyReLU(alpha=0.1)(blockInput)
    if batchnorm:
        x = BatchNormalization()(x)
        blockInput = BatchNormalization()(blockInput)
    x = convolution_block(x, num_filters, (3,3), batchnorm)
    x = convolution_block(x, num_filters, (3,3), batchnorm, activation=False)
    x = Add()([x, blockInput])
    return x

def cut_output(x):
    shapes = x.shape.as_list()
    new_h_shape = shapes[1] - 24
    new_w_shape = shapes[2] - 28
    return tf.slice(x, [0,12,14,0], [-1, new_h_shape, new_w_shape, -1])
   

def define_model(paramDict):
    input_size =  (paramDict['input_size'][0],paramDict['input_size'][1],paramDict['input_size'][2])
    
    # Create new input layer that adds padding to the input
    input_layer = Input(input_size, name='Padded_input')  
    input_pad = ZeroPadding2D(padding=(12,14), name="Input_zeropad")(input_layer) 

    # First create unet ++ from resnet_50, and then connect to the zeropadded input
    input_size_pad =  (paramDict['input_size'][0]+24,paramDict['input_size'][1]+28,paramDict['input_size'][2])
    backbone = tf.keras.applications.resnet50.ResNet50(include_top=False,
                          weights=paramDict['init_weights'],
                          input_shape=input_size_pad,
                          pooling=None)

    start_neurons = 8
    dropout_rate = paramDict['dropout_rate']

    conv4 = backbone.get_layer("conv4_block6_out").output 
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4) 
    pool4 = Dropout(dropout_rate)(pool4)
    
    # Middle
    convm = Conv2D(start_neurons * 32, (3, 3), activation=None, padding="same",name='conv_middle')(pool4)
    convm = residual_block(convm, paramDict['batchnorm'], start_neurons * 32)
    convm = residual_block(convm, paramDict['batchnorm'], start_neurons * 32)
    convm = LeakyReLU(alpha=0.1)(convm)
    
    deconv4 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(convm)
    deconv4_up1 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(deconv4)
    deconv4_up2 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(deconv4_up1)
    deconv4_up3 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(deconv4_up2)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(dropout_rate)(uconv4) 
    
    uconv4 = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4, paramDict['batchnorm'], start_neurons * 16)
    uconv4 = LeakyReLU(alpha=0.1)(uconv4) 
    
    deconv3 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv4)
    deconv3_up1 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(deconv3)
    deconv3_up2 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(deconv3_up1)
    conv3 = backbone.get_layer("conv3_block4_out").output
    uconv3 = concatenate([deconv3,deconv4_up1, conv3])    
    uconv3 = Dropout(dropout_rate)(uconv3)
    
    uconv3 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3, paramDict['batchnorm'], start_neurons * 8)
    uconv3 = LeakyReLU(alpha=0.1)(uconv3)

    deconv2 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv3) 
    deconv2_up1 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(deconv2) 
    conv2 = backbone.get_layer("conv2_block3_out").output 
    uconv2 = concatenate([deconv2,deconv3_up1,deconv4_up2, conv2])
        
    uconv2 = Dropout(dropout_rate/2)(uconv2)
    uconv2 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2, paramDict['batchnorm'], start_neurons * 4)
    uconv2 = LeakyReLU(alpha=0.1)(uconv2)
    
    deconv1 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv2) 
    conv1 = backbone.get_layer("conv1_relu").output 
    uconv1 = concatenate([deconv1,deconv2_up1,deconv3_up2,deconv4_up3, conv1])
    
    uconv1 = Dropout(dropout_rate/2)(uconv1)
    uconv1 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv1)  
    uconv1 = residual_block(uconv1, paramDict['batchnorm'], start_neurons * 2)
    uconv1 = LeakyReLU(alpha=0.1)(uconv1)
    
    uconv0 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv1)  
    uconv0 = Dropout(dropout_rate/2)(uconv0)
    uconv0 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv0)
    uconv0 = residual_block(uconv0, paramDict['batchnorm'], start_neurons * 1)
    uconv0 = LeakyReLU(alpha=0.1)(uconv0)
    
    uconv0 = Dropout(dropout_rate/2)(uconv0)
    output_layer = Conv2D(paramDict['num_classes'], (1,1), padding="same", activation=None)(uconv0)    
    output_layer = Softmax()(output_layer)

    # Cut the rows and cols again the make sure the output size matches the size of the targets    
    output_layer = Lambda(lambda x:cut_output(x))(output_layer)
    print('output layer shape: ', output_layer.shape)

    # Generate the resnet
    unet_plusplus = Model(backbone.input, output_layer)

    final_output = unet_plusplus(input_pad) # Feed the zero-padded input    
    model = Model(input_layer, final_output)

    model._name = 'resnet50_unet_plusplus'

    return model


def model_init(paramDict):
    model = define_model(paramDict)
    return model

