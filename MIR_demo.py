'''
    Name: C. Liu & K. Pareek
    Subject: Music Information Retrieval (MPATE-GE 2623)
'''

import os
import scipy
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ml_process import SpeechDetection,DataFrameSelector

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC #Kernel Trick
from sklearn.linear_model import SGDClassifier #SGD
from sklearn.naive_bayes import GaussianNB 
from sklearn.linear_model import LogisticRegression #Logistic
from sklearn.pipeline import Pipeline


#==============================1.Setup the Parameters=========================================
#Create dic as struct
params = {
        'downsampling':22050,
        'win_size': 1024,
        'hop_size': 512,
        'min_freq': 80,
        'max_freq': 8000,
        'num_mel_filters': 128,
        'n_dct': 20}

#Used Attributes for training estimator, which means it should drop the label, ID, and other irrelevent info
Training_Attributes = ['Short Time Energy',
                        'Average Energy',
                        'Low Short Time Energy Ratio',
                        'ZCR',
                        'Norm ZCR',
                        'Spectral Flatness',
                        'Spectral Centroid',
                        'Spectral Rolloff', 
                        'Spectral Entropy',
                        'MFCC_Cof_1', 'MFCC_Cof_2', 'MFCC_Cof_3', 'MFCC_Cof_4', 
                        'MFCC_Cof_5', 'MFCC_Cof_6', 'MFCC_Cof_7', 'MFCC_Cof_8', 
                        'MFCC_Cof_9', 'MFCC_Cof_10', 'MFCC_Cof_11', 'MFCC_Cof_12',
                        'MFCC_Cof_13', 'MFCC_Cof_14','MFCC_Cof_15', 'MFCC_Cof_16',
                        'MFCC_Cof_17', 'MFCC_Cof_18', 'MFCC_Cof_19', 'MFCC_Cof_20',
                        'PCEN_MFCC_1', 'PCEN_MFCC_2', 'PCEN_MFCC_3', 'PCEN_MFCC_4', 
                        'PCEN_MFCC_5', 'PCEN_MFCC_6', 'PCEN_MFCC_7', 'PCEN_MFCC_8', 
                        'PCEN_MFCC_9', 'PCEN_MFCC_10', 'PCEN_MFCC_11', 'PCEN_MFCC_12',
                        'PCEN_MFCC_13', 'PCEN_MFCC_14','PCEN_MFCC_15', 'PCEN_MFCC_16',
                        'PCEN_MFCC_17', 'PCEN_MFCC_18', 'PCEN_MFCC_19', 'PCEN_MFCC_20']
#The Pipeline for clean the dataframe
'''
Usually includes:
        *Fill F/A
        *Imputer
        *Handling the Text or other Category attibutes which notiing related to the real features
'''
DataCleaner = Pipeline([
                        ("feature_map", DataFrameSelector(Training_Attributes)),
                        ])
clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=75, n_jobs=1,
                oob_score=True, random_state=None, verbose=0,
                warm_start=False)

#=============================2.Setup the directory==========================================
#dir_path = os.getcwd() #Get Current Working Directory
speech_train_dir = os.getcwd() + '/datasets/speech'          #<<========Speech directory
soundscape_train_dir = os.getcwd() + '/datasets/soundscape'  #<<========Soundscape directory
#========================================Testing Annotated Audio==========================================
test_path = os.getcwd()+ '/datasets/perturbations' 
test_file = test_path + '/Noisy_5_15_25_40_50.wav'

SpeechDetection(Parameters=params, #The parameters for pre-processing/feature extraction
                TestFile=test_file, # The file path of the audio for detecting speech
                CleanPip=DataCleaner,
                Kfold=10,
                NewEstimator = True, 
                Estimator = clf,
                TestFile_Dir=test_file,
                Speech_Data = None,#'Mozilla15K_Speech_Data.csv', 
                NonSpeech_Data = None, #'UrbanSound6K_NonSpeech_Data.csv'
                Speech_Dir=None, # The Directory of Raw Speech audio files for generating dataset
                NonSpeech_Dir=None, #The Directory of Soundscape audio files for generating dataset
                WriteCSV=True #If True, the program will export a .csv file for saving the readed dataset
)
