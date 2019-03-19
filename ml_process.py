'''
    Name: C. Liu & K. Pareek
    Subject: Music Information Retrieval (MPATE-GE 2623)
'''

import numpy as np
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy
import pandas as pd


from library import read_audio
from library import Avg_Energy
from library import LSTER #Low short time energy ratio
from library import HZCRR #High zero crossing rate ratio
from library import PCEN_MFCC # Common MFCCs and MFCC with PCEN scaling
from library import Spectral_Entropy #Spectral Entropy
from library import ShortTimeEnergy #short Time Energy
from ml_utilities import plot_confusion_matrix

from sklearn.externals import joblib # For saving classifiers


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer #Used to take care of missing values in data
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_curve, auc
from scipy import interp
from joblib import dump, load

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
#================================================================================================
def dataset_processor(target_dataset,class_label,files_count,params):
    #If process raw audio files for the first time, set files_count = 0
    #Batch Processing
    win_size = params['win_size']
    hop_size = params['hop_size']
    fmin = params['min_freq']
    fmax = params['max_freq']
    n_mels = params['num_mel_filters']
    n_dct = params['n_dct']
    downsampling = params['downsampling']
    feature_set_flag = 0
    group_index = files_count + 1
    for filename in os.listdir(target_dataset):
        if filename.endswith('.mp3') or filename.endswith('.wav'):      
            print(filename) 
            print(group_index)
            file_path = target_dataset + '/' + filename        
            #================================Load Audio(Dowmsampled&Mono)=========================
            y,sr = librosa.load(file_path,sr = downsampling, mono = True, offset=1.0)        
            #================================STFT(Framing&windowing)==============================
            y_mag = np.abs(librosa.stft(y, n_fft=win_size, hop_length=hop_size))       
            #============================Feature computing===========================================
            #****************************************************************************************           
            #                                   Update                                              *
            #                           Features-Extraction Algorithm                               *
            #                                    Below                                              *
            #****************************************************************************************
            #Short Time Energy
            ste = ShortTimeEnergy(signal=y,parameters = params)
            #Average Energy in X seconds window(Defined in seconds long)
            ave_energy = Avg_Energy(signal_mag=y_mag,sr=sr,parameters=params,seconds=1) 
            #Low Short Time Energy Ratio
            lster = LSTER(signal_mag=y_mag,avg_pow=ave_energy,sr=sr,threshold=1)
            #ZCR
            zcr = librosa.feature.zero_crossing_rate(y=y,frame_length=win_size,hop_length=hop_size)
            #HZCRR
            hzcrr = HZCRR(zcr=zcr,sr=sr,parameters=params,seconds=1,threshold=1.5)
            #Spectral Flatness
            flatness = librosa.feature.spectral_flatness(y=y,n_fft=win_size, hop_length=hop_size)
            #Spectral Centroid
            cent = librosa.feature.spectral_centroid(y=y, sr=sr,n_fft=win_size, hop_length=hop_size)
            #Spectral Entropy
            entropy = Spectral_Entropy(y_Mag=y_mag,sr=sr,n_short_blocks=10)
            #Spectral Rolloff
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr,n_fft=win_size, hop_length=hop_size)
            #MFCCs and MFCCs with PCEN scaling
            mfccs, pcens = PCEN_MFCC(y=y,sr=sr,parameters=params)

            #Feature&Label Matrix building for the first time processing
            if feature_set_flag == 0:
                feature_set_flag = 1
                feature_ste = ste
                feature_ave_energy = ave_energy
                feature_lster = lster
                feature_zcr = zcr
                feature_hzcrr = hzcrr
                feature_flatness = flatness
                feature_cent = cent
                feature_rolloff = rolloff
                feature_entropy = entropy
                feature_mfccs = mfccs
                feature_pcens = pcens
                file_index = np.ones(y_mag.shape[1]) * group_index
                group_index = group_index + 1
            else:
                new_file_idex = np.ones(y_mag.shape[1]) * group_index
                file_index = np.concatenate([file_index,new_file_idex])
                feature_ste = np.append(feature_ste,ste)
                feature_ave_energy = np.append(feature_ave_energy,ave_energy)
                feature_lster = np.append(feature_lster,lster)
                feature_zcr = np.append(feature_zcr,zcr)
                feature_hzcrr = np.append(feature_hzcrr,hzcrr)
                feature_flatness = np.append(feature_flatness,flatness)
                feature_cent = np.append(feature_cent,cent)
                feature_rolloff = np.append(feature_rolloff,rolloff)
                feature_entropy = np.append(feature_entropy,entropy)
                feature_mfccs = np.concatenate([feature_mfccs,mfccs],axis = 1)
                feature_pcens = np.concatenate([feature_pcens,pcens],axis = 1)
                group_index = group_index + 1
    
    #==============================out of loop=========================
    label_col = np.ones(feature_mfccs.shape[1]) * class_label
    raw_data = np.concatenate([np.asarray([label_col]).T,
                                    np.asarray([file_index]).T,
                                    np.asarray([feature_ste]).T,
                                    np.asarray([feature_ave_energy]).T,
                                    np.asarray([feature_lster]).T,
                                    np.asarray([feature_zcr]).T,
                                    np.asarray([feature_hzcrr]).T,
                                    np.asarray([feature_flatness]).T,
                                    np.asarray([feature_cent]).T,
                                    np.asarray([feature_rolloff]).T,
                                    np.asarray([feature_entropy]).T,
                                    feature_mfccs.T,
                                    feature_pcens.T
                                                #####################
                                                #Other features here#
                                                #####################
                                                ],axis = 1)
    files_count = group_index
    return raw_data, files_count
#=================================================================================================================
def creat_set(speech_dir, nonspeech_dir,params):
    speech_data,speech_files_num = dataset_processor(speech_dir,class_label=1,files_count = 0,params=params)
    nonspeech_data,total_files_num = dataset_processor(nonspeech_dir,class_label=0,files_count = speech_files_num - 1,params=params)
    #Making the pandas dataframe
    speech_df = pd.DataFrame(speech_data)
    nonspeech_df = pd.DataFrame(nonspeech_data)
    columns_list = ['Label',
                        'FileID',
                        'Short Time Energy',
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
    speech_df.columns = columns_list
    nonspeech_df.columns = columns_list
    return speech_df,nonspeech_df
#=================================================================================================================

def Audio_Predict(file_dir,clf,parameters,pipeline):
    win_size = parameters['win_size']
    hop_size = parameters['hop_size']
    fmin = parameters['min_freq']
    fmax = parameters['max_freq']
    n_mels = parameters['num_mel_filters']
    n_dct = parameters['n_dct']
    downsampling = parameters['downsampling']
    y,sr = librosa.load(file_dir ,sr = downsampling, mono = True) #<<change parameter later
    y_mag = np.abs(librosa.stft(y, n_fft=win_size, hop_length=hop_size))  
    y_pow = librosa.amplitude_to_db(y_mag, ref=np.max)

    #Short Time Energy
    ste = ShortTimeEnergy(signal=y,parameters = parameters)
    #Average Energy in X seconds window(Defined in seconds long)
    ave_energy = Avg_Energy(signal_mag=y_mag,sr=sr,parameters=parameters,seconds=1) 
    #Low Short Time Energy Ratio
    lster = LSTER(signal_mag=y_mag,avg_pow=ave_energy,sr=sr,threshold=1)
    #ZCR
    zcr = librosa.feature.zero_crossing_rate(y=y,frame_length=win_size,hop_length=hop_size)
    #HZCRR
    hzcrr = HZCRR(zcr=zcr,sr=sr,parameters=parameters,seconds=1,threshold=1.5)
    #Spectral Flatness
    flatness = librosa.feature.spectral_flatness(y=y,n_fft=win_size, hop_length=hop_size)
    #Spectral Centroid
    cent = librosa.feature.spectral_centroid(y=y, sr=sr,n_fft=win_size, hop_length=hop_size)
    #Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr,n_fft=win_size, hop_length=hop_size)
    #Spectral Entropy
    entropy = Spectral_Entropy(y_Mag=y_mag,sr=sr,n_short_blocks=10)
    #MFCCs and MFCCs with PCEN scaling
    mfccs, pcens = PCEN_MFCC(y=y,sr=sr,parameters=parameters)
    audioData = np.concatenate([np.asarray([ste]).T,
                                np.asarray([ave_energy]).T,
                                np.asarray([lster]).T,
                                np.asarray(zcr).T,
                                np.asarray([hzcrr]).T,
                                np.asarray(flatness).T,
                                np.asarray(cent).T,
                                np.asarray(rolloff).T,
                                np.asarray([entropy]).T,
                                mfccs.T,
                                pcens.T],axis = 1)
    print('Size of Audio Data: {} '.format(audioData.shape))
    audioData_df = pd.DataFrame(audioData)
    audioData_df.columns = [
                        'Short Time Energy',
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
    print("Predicting the prpbability on testing audio...")
    predicted_label_bool = []
    predicted_proba_list = [] 
    
    audioData_np = pipeline.transform(audioData_df)
    for i in range(audioData_np.shape[0]):
        test_frame = audioData_np[i,:]
        #Predict Class
        temp_label = clf.predict([test_frame])#<<========================ChangeClassifier here
        #Predict Probability
        temp_proba = clf.predict_proba([test_frame])[0][1] #[0][0] is non-speech; [0][1] is speech
        predicted_label_bool.append(temp_label)
        predicted_proba_list.append(temp_proba)
    predicted_label = np.asarray(predicted_label_bool).astype(int)
    predicted_proba = np.asarray(predicted_proba_list)
    print('...Done!\n')
    #=================================================================================================================
    t_wave = np.linspace(0.0,len(y)/sr,len(y))
    t_stft= np.linspace(0.0,len(y)/sr,np.ceil(len(y)/hop_size))
    plt.figure(3,figsize = (15,25))
    plt.subplot(15,1,1)
    plt.plot(t_wave,y)
    plt.title('Waveform')
    
    plt.subplot(15,1,2)
    plt.pcolormesh(y_pow)
    plt.title('Power Spectrogram')
    
    plt.subplot(15,1,3)
    plt.plot(t_stft,ste)
    plt.title('Short Time Energy')
    
    plt.subplot(15,1,4)
    plt.plot(t_stft,ave_energy)
    plt.title('Average Energy in 1 Second-Window')
    
    plt.subplot(15,1,5)
    plt.plot(t_stft,lster)
    plt.title('Low Short Time Energy Ratio')
    
    plt.subplot(15,1,6)
    plt.plot(t_stft,zcr.ravel())
    plt.title('Zero Crossing Rate')
    
    plt.subplot(15,1,7)
    plt.plot(t_stft,hzcrr)
    plt.title('High Zero Crossing Rate Ratio')
    
    plt.subplot(15,1,8)
    plt.plot(t_stft,flatness.ravel())
    plt.title('Spectral Flatness')
    
    plt.subplot(15,1,9)
    plt.plot(t_stft,cent.ravel())
    plt.title('Spectral Centroid')
    
    plt.subplot(15,1,10)
    plt.plot(t_stft,rolloff.ravel())
    plt.title('Spectral Rolloff')

    plt.subplot(15,1,11)
    plt.plot(t_stft,entropy)
    plt.title('Spectral Entropy')   
    
    plt.subplot(15,1,12)
    librosa.display.specshow(mfccs, x_axis='time')
    plt.title('MFCCs')
    
    plt.subplot(15,1,13)
    librosa.display.specshow(pcens, x_axis='time')
    plt.title('PCEN-MFCCs')
    
    plt.subplot(15,1,14)
    plt.plot(t_stft,predicted_label)
    plt.title('Predicted Label')
    
    plt.subplot(15,1,15)
    plt.plot(t_stft,predicted_proba)
    plt.title('Predicted Probability')
    print("Time Vector Shape: {}".format(t_stft.shape))
    print("Probability Vector Shape: {}".format(predicted_proba.shape))
    
    plt.tight_layout()
    plt.savefig("Testing Audio Plot.png", dpi=150)
    plt.show()

    return predicted_label,predicted_proba,t_stft
    #------------------------------------------------------------------------------------------------
def SpeechDetection(Parameters,
                    TestFile,
                    CleanPip,
                    Kfold,
                    NewEstimator = False,
                    Estimator = None, 
                    TestFile_Dir = None,
                    Speech_Data = None,
                    NonSpeech_Data = None, 
                    Speech_Dir = None, 
                    NonSpeech_Dir = None, 
                    WriteCSV = False):

    #Using persistence .csv dataset
    if Speech_Data and NonSpeech_Data:
        speech_df = pd.read_csv(Speech_Data)
        nonspeech_df= pd.read_csv(NonSpeech_Data)
        print('Read {} Successfully!'.format(Speech_Data))
        print('Read {} Successfully!'.format(NonSpeech_Data))
    else:
    #Create dataset by processing all the raw audio files
        if Speech_Dir == None or NonSpeech_Dir == None:
            raise ValueError('\nPlease Offer a .csv Dataset, or Assign the Paths of Your Speech and Soundscape Audio Files for Generating the Dataset')
        speech_df,nonspeech_df = creat_set(speech_dir = Speech_Dir,
                                            nonspeech_dir = NonSpeech_Dir,
                                            params = Parameters)
        if WriteCSV:
            speech_df.to_csv('Speech_Data.csv', index=False)
            nonspeech_df.to_csv('NonSpeech_Data.csv', index=False)
    print("Size of Speech set: {}".format(speech_df.shape)) 
    print("Size of Soundscape set: {}".format(nonspeech_df.shape))
    #======================================================================================================
    #======================================================================================================
    print("Combining the training sets of two classes...")
    train_set = pd.concat([speech_df, nonspeech_df],sort=False)
    train_set.to_csv('testdemocsv.csv', index=False)
    print("Size of total set {}".format(train_set.shape))
    #Shuffle all training set
    train_set = train_set.sample(frac=1).reset_index(drop=True)
    '''
    #Drop label and ID colomun
    train_set = train_set.drop(columns=['Label','FileID']) #could be moved in pipeline/Replace by Column selector
    #Convert dataframe to numpy array
    train_set_np = train_set.values #could be moved in pipeline
    '''
    plt.figure(1,figsize = (15,15)) # For Confusion Matrix
    plt.figure(2,figsize = (15,15)) # For ROC curve
    #==============================================
    class_names = ['Non-Speech', 'Speech']  #0 = Non-Speech, 1 = Speech
    #Prepare saving data from the iteratoring Kfold
    subplot_index = 1
    subplot_col = np.floor(np.sqrt(Kfold))
    subplot_row = np.ceil(Kfold/subplot_col)
    
    cv_score = []
    clf_accuracy_score = []
    recall = [] #Store recall score in each fold
    f1 = [] #Store f1 score in each fold
    precision = []  #Store precision score in each fold
    tprs = []   #Store tpr  in each fold
    aucs = []   #Store auc score in each fold
    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    fold_idx = 0
    #==============================================
    #Gourps here is for GroupKFold to avoid the frames of same audio clip split into both train and test set
    groups = train_set['FileID'].tolist()
    groups = np.array([int(i) for i in groups]) #Convert to int
    #Setup the n_splits number for times of folding
    group_kfold = GroupKFold(n_splits=Kfold)
    #True for all the data with label 1
    speech_labels = train_set["Label"]
    train_set_np = CleanPip.transform(train_set)

    model_path = os.getcwd()+ '/models' 
    #Using persistence models
    if NewEstimator == False:
        #Load persistence estimators
        for filename in os.listdir(model_path):
            if filename.endswith('.pkl'):
                clf_file = model_path + '/' + filename
                clf = load(clf_file) 
                print('Reading persistence model {}...'.format(filename))
    #Using new estimators
    else:
        if Estimator == None:
            raise ValueError('\nPlease Offer a valid estimator instance for the parameter "Estimator"')
        else:
            clf = Estimator
    #==========================================================================================
    #========================KGroup-Folding Validation=====================================
    for train_index, test_index in group_kfold.split(train_set_np, speech_labels, groups):
        fold_idx += 1
        print('\n==========================Fold.{}========================'.format(fold_idx))
        X_train, X_test = train_set_np[train_index], train_set_np[test_index]
        y_train, y_test = speech_labels[train_index], speech_labels[test_index]
        '''
        May use Train set to fit pipeline and transform tranin and test set here
        scaler = StandardScaler().fit(X_train)
        X_train_std = scaler.transform(X_train)
        X_test_std = scaler.transform(X_test)
        clf.fit(X_train_std,y_train)
        scores = cross_val_score(clf, X_test_std, y_test, cv=0)
        '''
        if NewEstimator == True:
            clf.fit(X_train,y_train)
        #scores = cross_val_score(clf, X_test, y_test, cv=1)
        cv_results = cross_validate(clf, X_test, y_test, cv=2, return_train_score=True)
        print('A sub cross-validation on test set:\n')
        print('Fit Time: {}\n'.format(cv_results['fit_time']))
        print('Score Time: {}\n'.format(cv_results['score_time']))
        print('Test Score: {}\n'.format(cv_results['test_score']))
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test) #1 indicates speech
        y_pred_posproba = y_pred_proba[:,1]
        #====================Confusion Matrix with acuuracy measurement====
        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_test, y_pred)
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure(1)
        plt.subplot(subplot_row,subplot_col,subplot_index)
        subplot_index += 1
        plot_confusion_matrix(cnf_matrix, classes=class_names,normalize=False,
                        title='Confusion matrix in {} fold, without normalization'.format(fold_idx))    
        #Measurement
        clf_accuracy_score.append(accuracy_score(y_test, y_pred))
        temp_recall = recall_score(y_test, y_pred)
        temp_precision = precision_score(y_test, y_pred)
        temp_f1 = f1_score(y_test, y_pred)
        recall.append(temp_recall)
        precision.append(temp_precision)
        f1.append(temp_f1)
        #======= Compute ROC curve and ROC area for each class==========
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_posproba, pos_label=1)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        print('Recall: {}\n'.format(temp_recall))
        print('Precision: {}\n'.format(temp_precision))
        print('F1: {}\n'.format(temp_f1))
        plt.figure(2)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1
        #===============================================================
    #Finish ROC curve
    plt.figure(2)    
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("ROC_Curve.png",dpi=150)
    plt.figure(1)
    plt.savefig("Confusion Matirx.png", dpi=150)
    if NewEstimator == True:
        dump(clf, model_path + '/RandomForestVoiceDetector.joblib') 
    #==============================Exit the machine learning area==================================
    if TestFile_Dir:
        predicted_label,predicted_proba,time_vector = Audio_Predict(TestFile_Dir,clf,Parameters,CleanPip)
        
