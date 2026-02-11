import tensorflow as tf
from datetime import datetime


def callbacks(filename,callbacks_list,path):

    """ 
        A function call for defining callbacks. Callbacks are tasks performed by neural network at different stages during its training phase.
            1.  ReduceLROnPlateau - Learning rate is lowered if no improvement in model performance i.e. model performance is plateaued. 
                                    "Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates. This callback monitors a quantity and if no improvement is seen for a 'patience' number of epochs, the learning rate is reduced."
            
            2. Early stopping callback - stops the execution of training epochs if the 'val_loss' parameter remains more or less constant and refuses to minimize further. 
                                        Useful where model training defines a higher number of epochs e.g. 50 but the training stops if there is no further improvement in 'val_loss.
            
            3. Model checkpoint - Saves the model and model weights in a specific file - so that the model could be loaded separately for verifying with test data.

            4. CSVLogger - Saves epoch results to a CSV file
    """

    if callbacks_list.__len__() > 0:
        callbacks_list = []

    reduceLR = tf.keras.callbacks.ReduceLROnPlateau(
        monitor = 'val_loss',
        factor=0.2,     # factor by which the learning rate has to be reduces
        patience=3,
        mode = 'min',
        min_lr=1e-6,
        verbose=1
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        patience = 5,   # if the val_loss remains monotonic for 5 epochs, then stop the execution
        mode = 'min',    # 'min' denotes the direction of monitoring measurement 'val_loss' i.e. ensures val_loss is decreasing
        restore_best_weights = True,     # saves optimal model weights i.e. the weights from the specific epoch that performed the best on validation set - regardless of the number of epochs executed
        verbose=1
    )

    model_checkpoint_auc = tf.keras.callbacks.ModelCheckpoint(
        filepath = f'{path}/auc/{filename}',
        monitor = 'val_auc',
        save_best_only = True,   # saves only the model that performs best in terms of 'val_loss'
        mode='max',
        verbose=1
    )
	
    model_checkpoint_loss = tf.keras.callbacks.ModelCheckpoint(
        filepath = f'{path}/loss/{filename}',
        monitor = 'val_loss',
        save_best_only = True,   # saves only the model that performs best in terms of 'val_loss'
        mode='min',
        verbose=1
    )

    csv_logger = tf.keras.callbacks.CSVLogger(
        f'{path}/history.csv',      # path to log file
        append=True
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=f'{path}/logs/logs_{datetime.now().strftime("%Y%m%d-%H%M%S")}', 
        histogram_freq=1, 
        write_graph=True,
        write_images=True,
        write_steps_per_second=True
    )

    callbacks_list = [reduceLR,early_stopping,model_checkpoint_auc,model_checkpoint_loss,csv_logger,tensorboard_callback]
    return callbacks_list



""" 
    Blocks of function calls to build a U-Net Neural Network architecture with skip connections.
"""

def conv_block(input,filters):
  
    """ 
        Function call to create the sequence of layers, like:
            conv2D --> conv2D -->  BN (Batch normalization) --> Activation 'ReLU'
    """
    convolution = tf.keras.layers.Conv2D(filters,(2,2),strides=(1,1),padding='same',kernel_initializer='he_normal')(input)       # convolution operration - shrinking spatial dimensions; 'he_normal' ...
    convolution = tf.keras.layers.Conv2D(filters,(2,2),strides=(1,1),padding='same',kernel_initializer='he_normal')(convolution)
    bn = tf.keras.layers.BatchNormalization()(convolution)
    convolution = tf.keras.layers.Activation('relu')(bn)
    return convolution

def build_encoder(input,filters):
    
    """ 
    Function call to create downsampling layers using MaxPooling2D.
    The function returns 2 parameters,
        1. convolution: features derived from convolution (saved at this stage, to be passed to corresponding upsampled layer later)
        2. downsampling: the output of current CNN layer to be passed on to the next layer as input

    """

    convolution = conv_block(input,filters)
    downsampling = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(convolution)      # pool size reduces the height and width by half - improving computational efficiency
    return convolution, downsampling


def build_decoder(input,filters,skip_features):
        
    """ 
        Function call to create upsampling layers using Conv2DTranspose. The layer structure here follows the pattern:
            conv2DTranspose --> feature concatenation --> conv2D --> conv2D --> BN (Batch Normalization) --> Activation 'ReLU'
        
        This function also combines the saved features during downsampling to the upsampled image (i.e. the ability of skip connections)
    """

    upsampling = tf.keras.layers.Conv2DTranspose(filters,(2,2),strides=(2,2),padding='same')(input)          # deconvolution operation - extending spatial dimensions
    combined = tf.keras.layers.Concatenate()([upsampling,skip_features])                                 # concatenating upsampled layer with original derived features
    convolution = tf.keras.layers.Conv2D(filters,(2,2),strides=(1,1),padding='same',kernel_initializer='he_normal')(combined)
    convolution = tf.keras.layers.Conv2D(filters,(2,2),strides=(1,1),padding='same',kernel_initializer='he_normal')(convolution)
    bn = tf.keras.layers.BatchNormalization()(convolution)
    convolution = tf.keras.layers.Activation('relu')(bn)
    return convolution
