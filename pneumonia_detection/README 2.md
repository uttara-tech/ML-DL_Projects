---
title: COVID-19 Radiography Database
emoji: 🐨
colorFrom: green
colorTo: green
sdk: gradio
sdk_version: 6.5.1
app_file: app.py
python_version: "3.12.12"
pinned: True
short_description: 'U-NET model for Multi-class classification '
---

The dataset for this project is downloaded from Kaggle - 'COVID-19 Radiography Database'. Available at: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database.

This is an experimental multi-class classification model, based on U-NET architecture, to help medical practioners in diagnosis of a lung disease, namely, 
1. Viral Pneumonia
2. COVID
3. Lung Opacity (that indicates a disease in the lung without specifying which one)
4. Normal

Notwithstanding the hardware restrictions - the model produces acceptable values for Precision, Recall and F1-score, although not accurate enough to be used in a medical setting yet.

# <font color='Maroon'> Steps for installaton</font>

## <font color='Teal'> [A] Setting up a VSCode workspace on Mac</font>

**Required Python version - 3.12.12**

1. Create a virtual environment with Python 3.12.12
2. VSCode prompts installing libraries from 'requirements.txt' - install all dependencies.
3. !! IMP !! Do not change the Tensorflow version. 
    
    Tensorflow 2.18.1 is compatible with Python 3.12.12. 
    
    Tensorflow-metal is essential for GPU acceleration on Apple Silicon chips (M1,M2,M3,M4).

4. Optionally, install "macmon" with Homebrew to monitor GPU and CPU utilization 
5. Go to ../pneumonia_detection/notebooks/

    a) Configure 'config.json' with the local path to your directory

    b) Execute pneumonia-detection.ipynb notebook
6. Dataset will be downloaded from Kaggle into folder ../pneumonia_detection/data/

Alternatively, export 'pneumonia-detection.ipynb' into a python script and run this script in a Terminal.


## <font color='Teal'> [B] Setting up VSCode workspace on Windows 11 system with NVIDIA hardware (GPU) acceleration</font>

Setting up AI environment on Windows is highly dependent on the hardware. The given steps are compatible with below hardware specifications.
  **OS**: Windows 11
  **GPU**: NVIDIA GeForce RTX 5090
  **Driver**: NVIDIA Studio Driver

**Required Python version - 3.10.8**

1. Create a virtual environment with Python 3.10.8

2. The dependencies for GPU acceleration:

    a) CUDA version 11.5 (for Windows 10 - 11.0)
    b) Tensorflow version 2.10.0 is compatible with the specifies python version
    c) Optionally: numpy<1.24, protobuf==3.19.6, tensorflow-datasets==4.6.0

3. Go to ../pneumonia_detection/notebooks/

    a) Configure 'config.json' with the local path to your directory

    b) Execute pneumonia-detection.ipynb notebook
4. Dataset will be downloaded from Kaggle into folder ../pneumonia_detection/data/

Alternatively, export 'pneumonia-detection.ipynb' into a python script and run this script in PowerShell. Monitor CPU and GPU utilization on Task Manager.


# <font color='Maroon'> Project directory structure</font>

### <font color='Teal'> |-- data </font>

This is where the dataset from Kaggle will be downloaded.

### <font color='Teal'> |-- models</font>

Contains all logs and model history - populated during training and retrieved for evaluating performance metrics. 

### <font color='Teal'> |-- notebooks</font>

Main Jupyter notebooks - 
- pneumonia-detection.ipynb, 
- performance_metrics.ipynb

Downsampling was performed in below notebooks to achieve practical trade-off between model performance and resource allocation -

- pneumonia-detection_downscaled.ipynb
- performance_metrics_downscaled.ipynb

Jupyter notebooks demonstrating -

    i.   Describing the problem statement
    ii.  Exploratory data analysis
    iii. Splitting the dataset
    iv.  Model training
    v.   Evaluating with Test dataset
    vi.  Performance metrics 

### <font color='Teal'> |-- scripts</font>

Custom python packages tailored for handling dataset 'COVID-19 Radiography Database'.
