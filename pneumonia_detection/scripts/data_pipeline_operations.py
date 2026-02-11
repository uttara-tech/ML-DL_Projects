import tensorflow as tf
import random

# target image dimesions
IMAGE_HEIGHT= 256        
IMAGE_WIDTH = 256  
IMAGE_CHANNELS = 1


def process_data_pipeline(img,msk,label):

    """    
        Processing data further by:
            1. image segementation is performed by applying bitwise AND between X-ray image pixels and Mask image pixels. This ensures that the neural network focuses only on the relevant region of the X-rays.
            2. finally normlization of this overlayed image is performed to scale all pixel values in the range 0-255 to a smaller, consistent range 0-1. This enables the model to genralize faster.
            3. Optionally, image augmentation could also be performed at this stage on the already segmented images.
    """
           

    img = tf.cast(img,tf.uint8)
    msk = tf.cast(msk,tf.uint8)

    # Creating an overlay by logically placing Mask images over X-ray images
    overlay = tf.bitwise.bitwise_and(img,msk) 
    overlay = tf.cast(overlay,tf.float16) 
                                  
                          
    overlay = overlay / 255.0                                     # image normalization - to scale all pixel values to a rane 0-1. Helps the model to generalize faster.

    return overlay, label


def image_augmentation(
                       flip,
                       rotation,
                       ):

    image_augmentor = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(flip),
    tf.keras.layers.RandomRotation(rotation)
    ])

    return image_augmentor


imgage_augmentation_layer = image_augmentation( flip='horizontal',
                                                rotation=0.1
                                        )
@tf.function
def preprocess_data_pipeline(inputs,label):


    """ 
        Input pipeline function call to pre-process the batched images while the neural network is executing prior batch of data.
        Tensorflow's core modules like tf.io for file-level operations and tf.image for image processing are used here to handle the lifecycle of an image.
            1. raw image is read from the file path
            2. raw image is decoded to its numerical equivalent in the form of a Tensor
            3. this tensor image is resized. Please note that if the available hardware is extensive, then this size could be increased further for better results
            4. Optionally, image augmentation could be performed here.
    """   

    try:
        img, msk = inputs
        # Pre-processing X-ray images
        img = tf.io.read_file(img)                 
        img = tf.image.decode_png(img,channels=IMAGE_CHANNELS)
        img = tf.image.resize(img,[IMAGE_HEIGHT,IMAGE_WIDTH])            # resizing the image to 64X64 for memory effciency. 
        
        # Pre-processing X-ray images
        msk = tf.io.read_file(msk)                           
        msk = tf.image.decode_png(msk,channels=IMAGE_CHANNELS)
        msk = tf.image.resize(msk,[IMAGE_HEIGHT,IMAGE_WIDTH])            # resizing the image to 64X64 for memory effciency. 

    except:
        raise Exception()


    return img,msk,label


def augment_data(overlay,label):  
    
    roll = random.uniform(0,1)
    if (roll > 0.5):
        overlay = imgage_augmentation_layer(overlay)

    return overlay, label

def create_ds(xrays, masks, labels):

    """ 
        Function call to create input pipelines using "tf.data.Dataset" API in Tensorflow's ecosystem. 
        This package allows us to implement highly efficient and scalable pipelines that can stream and transform data during runtime.
    """
    try:

        print('Creating dataset ...')
        dataset = tf.data.TFRecordDataset.from_tensor_slices(((xrays,masks),labels))    # Load data from memory
        print('The tensor specifications for 2 features "X-rays & Masks" and 1 target variable "Labels" are: \n',dataset.element_spec)    # 3 Tensor specifications are created representing each input data element
        dataset = (dataset
            
            .map(preprocess_data_pipeline,num_parallel_calls=tf.data.AUTOTUNE)      # pre-processing data before - resizing and normalizing
            
            .map(process_data_pipeline,num_parallel_calls=tf.data.AUTOTUNE))     # overlaying X-rays with their corresponding masks
        
            #.map(augment_data,num_parallel_calls=tf.data.AUTOTUNE))     # Optional: image augmentation pipeline 
                
        print('\nThe tensor specifications after pre-processing: \n',dataset.element_spec,'\n')
        
    except:
        raise Exception()
    
    return dataset


def batch_dataset(dataset,batch_size,path):
    """ 
        Function call to create batches for train, validation and test datasets - after the split, to increases input pipeline performance.
    """

    try:

        dataset = dataset.batch(batch_size, drop_remainder=True)                                 # first time the dataset is iterated over, its elements will be cached in memory
        #dataset = dataset.cache(path)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)                            # enables preparation of next set of data while the current data is being processed
    
    except:
        raise Exception()
    
    return dataset


def split_dataset(ds):
    """
        Splitting the dataset in train, validation and test sets
    """

    ds_size = ds.cardinality().numpy()              # Total number of samples. ds_size = np.int64(2117)
    
    ds_train_size = int(ds_size * 0.7)              # 70% data to be reserved for training
    ds_val_size = int(ds_size * 0.1)                # 10% data to be reserved for validation
    ds_test_size = int(ds_size * 0.2)               # 20% data to be reserved for test

    train_ds = ds.take(ds_train_size)
    remaining = ds.skip(ds_train_size)

    test_ds = remaining.take(ds_test_size)
    remaining = remaining.skip(ds_test_size)
    val_ds = remaining.take(ds_val_size)

    return train_ds,val_ds,test_ds