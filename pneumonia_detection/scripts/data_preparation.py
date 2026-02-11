import tensorflow as tf
import numpy as np


def prepare_dataset(dict):
    """
        Function call to split data points into X-rays and Masks - and transforming the categorical label to its numerical equivalent
    """

    xrays = []
    masks = []
    labels = []

    for i, key in enumerate(dict): 
        for imgs in enumerate(dict[key]):
            _, img = [f for f in imgs]
            xrays.append(img[0])
            masks.append(img[1])
            labels.append(i)

    return xrays, masks, labels


def one_hot_encoding(labels):
    """ 
        Function call to convert categorical/numerical labels into their binary equivalent form.
    """

    binary_vectors = tf.keras.utils.to_categorical(labels,num_classes=4)
    print('LABELS: ',binary_vectors.shape)

    return binary_vectors


def display_unique_labels(labels,classes):
    """ 
        Function call to list unique binary vectors corresponding to labels
    """
    unique_vectors = np.unique(labels, axis=0)
    class_names = list(classes.keys())

    for vec in unique_vectors:
        label_index = np.argmax(vec)
        print(f"Vector {vec} corresponds to Class Index: {label_index} and original label as {class_names[label_index]}")


