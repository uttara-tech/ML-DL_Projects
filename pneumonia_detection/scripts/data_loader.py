
import os
import warnings
import kagglehub
import pathlib
from pathlib import Path
import tensorflow as tf
from collections import defaultdict


warnings.filterwarnings('ignore')



def load_dataset(dataset_name,dataset_download_dir):

    root_dir=''
    if not os.path.exists(dataset_download_dir):
        raise FileNotFoundError('Please check if the path is correct or appropriate access permissions are granted.')
    os.environ['KAGGLEHUB_CACHE']=dataset_download_dir
    
    exists_check = f'{dataset_download_dir}/COVID-19_Radiography_Dataset'
    if os.path.exists(exists_check):
        print('Dataset already  downloaded.')
        root_dir = exists_check
    else:
        path = kagglehub.dataset_download(dataset_name)
        print(f'Dataset downloaded successfully at: {path}')
        root_dir = os.path.join(path,'COVID-19_Radiography_Dataset')
        print('The root directory: ',root_dir)

    return root_dir

def get_classes(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Directory not found: {path}')
    print('Retrieveing files from: ', path)

    classes = defaultdict(list)
    print('Extracting class names ...')
    class_names = sorted([file for file in os.listdir(path) if not file.startswith('.') and not file.endswith(('.xlsx','.txt'))])
    print(f'There are {len(class_names)} available classes:')
    if os.path.exists(f' {path}/{class_names[0]}'):
        print("Files in this folder:", os.listdir(f'{path}/{class_names[0]}'))

    for i, cls_name in enumerate(class_names):
        print(f'\n {i+1}. {cls_name} ')
        sub_dir = [file for file in os.listdir(f'{path}/{cls_name}') if not file.startswith('.')]
        key = cls_name.upper().replace(' ', '_')
        classes[key] = []
        for j in sub_dir:
            print(f'    |-- {j}')
            classes[key].append(f'{path}/{cls_name}/{j}')
    
    return classes


def get_data(dict,num_samples=0):
    # Function call to extract a small sample of X-ray and Mask images with their respective classes

    samples = defaultdict(list)

    if num_samples > 0:
        for key in dict:
            x_path , m_path = dict[key]
            x_images = os.listdir(x_path)
            m_images = os.listdir(m_path)
            samples[key]=[]
            for (x, m) in zip(x_images[:num_samples],m_images[:num_samples]):
                if x.lower().endswith(('.png','.jpeg','.jpg')) and m.lower().endswith(('.png','.jpeg','.jpg')):
                    tup = (f'{x_path}/{x}',f'{m_path}/{m}')
                    samples[key].append(tup)
                else:
                    raise Exception(f'Unrecognised file format:{x} or {m}')
    else:
        for key in dict:
            x_path , m_path = dict[key]
            x_images = os.listdir(x_path)
            m_images = os.listdir(m_path)
            samples[key]=[]
            for (x, m) in zip(x_images,m_images):
                if x.lower().endswith(('.png','.jpeg','.jpg')) and m.lower().endswith(('.png','.jpeg','.jpg')):
                    tup = (f'{x_path}/{x}',f'{m_path}/{m}')
                    samples[key].append(tup)
                else:
                    raise Exception(f'Unrecognised file format:{x} or {m}')
    
    return samples

