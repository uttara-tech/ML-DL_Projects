# README

## Overview

### A brief outline of machine learning and deep learning projects:

### 1.	Atmospheric Emissions Prediction 

**Objective**: XGBoostRegression model for predicting CO2 emissions

**Libraries**: XGBoostRegressor (from xgboost), SHAP,mean_squared_error, Numpy, Pandas, Matplotlib, XGBoost, Seaborn

**Overview:**

a. Data preprocessing - checking for null values, replacing zero values with column mean

b. Feature Engineering for selecting most influencial components like extracting NOx, PM2.5, PM10 as features and CO2 as target.

c. Dataset filtering - extracting records only for the year 2019.

d. Used XGBoostRegressor for making predictions. A detailed report avaialble in the project folder.


### 2.	Student Information System

**Objective**: Data cleaning and pre-processing

**Libraries**: SimpleImputer, CountVectorizer, WordCloud, regex, missingno, LogisticRegression, OneHotEncoder, ColumnTransformer, Pipeline, LabelEncoder, Numpy, Pandas, Matplotlib, Seaborn

**Overview:**

The provided data had multiple issues. Initial observations (which are re-iterated in the jupyter file) are as below: 

a. Null values (multiple columns contained 'NaN' values)

b. Data accuracy and consistency issues (e.g. mixture of alphanumeric and integer values in 'ID' column)

c. Data uniformity issues (inconsistent column naming convention)

d. Noisy Data (e.g. special characters in some columns)

Below data cleaning measures were employed to make the data complete and valid by filling in missing values, followed by making it accurate, consistent, and uniform (detailed in Jupyter Notebook):

a. Missing Completely at Random (MCAR) 

b. Missing at Random (MAR) 

c. Missing Not at Random (MNAR)

d. Deletion and Imputaion


### 3.	Wine data analysis

**Objective**: Hyperparameter Tuning and Pipelining using XGBoost

**Libraries**: Numpy, Pandas, Matplotlib, XGBoost, Seaborn

**Overview:**

a. Data preprocessing - checking for null values

b. Data analysis and representation using Matplotlib.

c. Hyperparameter tuning and Pipelining using XGBoost.

d. Comparison of accuracy derived by different hyperparameter combinations using Grid Search, Random Search and Bayesian Optimization.

### 4.	Breast Cancer Prediction

**Objective:** Multi-class model in Machine Learning

**Libraries:** Scikit-learn, Numpy, Pandas, Matplotlib, Seaborn

**Overview:**

a. Pre-processing data, like checking for null values and drop unwanted columns, and standardizing all values using Scikit-learn (sklearn)

b. Performing PCA for dimensionality reduction. 10 out of 30 columns are enough to explain 95% of variance.

c. Training and Testing the data with these 10 columns.

d. Utilizing default classifiers for below machine learning algorithms:

    i)   Descision Tree Classifier
    ii)  Random Forest Classifier
    iii) XGB (eXtreme Gradient Boosting) Classifier
    iv)  Gradient Boosting Classifier
    v)   Extra Trees Classifier
    vi)  K-nearest Neighbors Classifier

e. Training the data with all the above classifiers

f. Testing the data, making predictions and calculating accuracy accordingly.

g. Using a confusion matrix to evalute the performance of all classifiers.

h. Representing and comparing all classifier performances with an AUC and ROC curve, and sklearn metrics like confusion matrix, accuracy score, precision, recall and F-score.

### 5.	Newton-Raphson

**Objective:** Newton-Raphson mathematical representation using Python

**Libraries:** SciPy, Math, Sympy, Numpy, Matplotlib

**Overview:**

a. Define a function using Math

b. Calculating derivative of the function to determine critical points
c. Using SciPy’s built-in Newton’s algorithm  for optimization of given function.
d. The code is a representation of mathematical calculations performed in Question (1.c): https://github.com/uttara-tech/typesetting-LaTeX/blob/8fbead2715a8d54141a1f6b09b3e073e6d3bd9a7/linear_algebra/Mid_Module_Assignment.pdf

    Repository - typesetting-LaTeX
    Folder - linear_algebra
    File - Mid_Module_Assignment.pdf, Question (1.c)

### 6.	Video Games sales data analysis

**Objective:** Exploratory Data Analysis

**Libraries:** Numpy, Pandas, Matplotlib, Seaborn

**Overview:**

a. Exploring and grouping the data.

b. Data visulalization using Matplotlib and Seaborn.

c. Used Heatmap, Histogram, Count plot, Bar plot, Group Bar plot, Joinplot.

### 7.	Students Performance

**Objective:** Exploratory Data Analysis

**Libraries:** Numpy, Pandas, Matplotlib, Seaborn

**Overview:**

a. Exploring and grouping the data.

b. Data visulalization using Matplotlib and Seaborn.

c. Used Distribution plot, Join plot, Pair plot, Box plot, Heatmap, Seaborn’s ‘catplot’ i.e.  plot showcasing relationship between numerical and categorical variables i.e. catplot, Count plot, Bar plot.

### 8.	[In progress] Sentiment Analysis

**Objective:** Prediction using RandomForrestClassifier and performance measure with sklearn metrics like confusion matrix, precision, recall, F-score, accuracy, auc and roc_curve

**Frameworks and Libraries:** cv2, InceptionV3, TensorFlow, Keras, Scikit-learn, Numpy, Pandas, Matplotlib, Seaborn

**Overview:**

### 9.	[In progress] Predict LoS (Length of Stay) of patients in a hospital

**Objective:** Prediction using Keras Model

**Frameworks and Libraries:** TensorFlow, Keras, Scikit-learn, Numpy, Pandas, Matplotlib, Seaborn

**Overview:**

a. Pre-processing data by:

    i) data cleaning, drop redundant columns and fill in missing values in cells

    ii) splitting data into numerical and categorical groups,

    iii) train and test dataset

b. Pre-process categorical features using:

    i) One Hot Encoding

    ii) Label Encoding

    iii) Tokenizer

    iv) Pre-process numeric data using convolution Neural Network for extracting features.

    v) Concatenate all inputs using a dictionary.

    vi) Pass this concatenated input to Keras Model.

    vii) Compile and Build the model


### 10.	[In progress] Transfer Learning using Inception

**Objective:** Prediction using Keras Model

**Frameworks and Libraries:** cv2, InceptionV3, TensorFlow, Keras, Scikit-learn, Numpy, Pandas, Matplotlib, Seaborn

**Overview:**


