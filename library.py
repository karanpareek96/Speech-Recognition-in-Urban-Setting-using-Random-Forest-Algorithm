'''
    Name: C. Liu & K. Pareek
    Subject: Music Information Retrieval (MPATE-GE 2623)
'''

import numpy as np
import os
import matplotlib.pyplot as plt
import librosa
import pandas as pd
import itertools
import scipy
from librosa.feature import melspectrogram

#=========================================================================================================
'''
*******************************************************************
Add Featrues Extraction Algorithms below, and updating the function in class_processor(), makedf_audio();
also you have to rearrange the order of attributes in create_set()
*******************************************************************
'''
def energy_sum(y_mag):
    y_pow = librosa.amplitude_to_db(y_mag)
    energy = np.zeros((y_pow.shape[1],1))
    for i in range(y_pow.shape[1]):
        energy[i] = sum(y_pow[:,i])
    return energy.T

def ShortTimeEnergy(signal,parameters):
    win_size = parameters['win_size']
    hop_size = parameters['hop_size']
    signal_mag = np.abs(librosa.stft(signal, n_fft=win_size, hop_length=hop_size)) 
    signal_pow = np.square(signal_mag)
    return np.sum(signal_pow, axis=0)/win_size #Short Time Energy,optional dived by win_size

def Avg_Energy(signal_mag,sr,parameters,seconds = 1):
    '''
    Calculate the 
        Avg_Energy: average short-time energy in a seconds-s window
    '''
    signal_pow = np.square(signal_mag)
    sum_pow = np.sum(signal_pow, axis=0)
    ave_pow = np.zeros(sum_pow.shape[0])
    ave_idx = 0
    
    one_second_len = int(np.ceil(seconds*sr/parameters['hop_size']))
    frame_start_index = 0
    
    while frame_start_index < sum_pow.shape[0]:
        frame_end_index = frame_start_index + one_second_len
        if frame_end_index > sum_pow.shape[0]:
            ave_pow[frame_start_index:-1] = np.sum(sum_pow[frame_start_index:-1])/(sum_pow.shape[0] - frame_start_index)
        else:
            ave_pow[frame_start_index:frame_end_index] = np.sum(sum_pow[frame_start_index:frame_end_index])/one_second_len
        frame_start_index = frame_end_index
        ave_idx = ave_idx +1
    return ave_pow

def LSTER(signal_mag,avg_pow,sr,threshold = 1):
    '''
    Calculate the 
        LSTER: the ratio of the number of frames whose
                STE are less than 'Threshold' time of average 
                short-time energy in a 1-s window(or other)
    '''
    signal_pow = np.square(signal_mag)
    sum_pow = np.sum(signal_pow, axis=0)
    return np.subtract(threshold*avg_pow,sum_pow)

def HZCRR(zcr,sr,parameters,seconds = 1,threshold = 1.5):
    '''
    Calculate the 
        High Zero-Crossing Rate Ratio: 
                HZCRR is defined as the ratio of the number of frames whose
                ZCR are above 1.5-fold average zero-crossing rate in an 1-s
                window
    '''
    average_window = int(np.ceil(seconds*sr/parameters['hop_size']))
    frame_start_index = 0
    zcr = np.ravel(zcr)
    hzcrr = np.zeros(zcr.shape)
    ave_zcr = 0
    ave_idx = 0
    while frame_start_index < zcr.shape[0]:
        frame_end_index = frame_start_index + average_window
        if frame_end_index > zcr.shape[0]:
            ave_zcr = np.sum(zcr[frame_start_index:-1])/(zcr.shape[0] - frame_start_index)
        else:
            ave_zcr = np.sum(zcr[frame_start_index:frame_end_index])/average_window
            
        #hzcrr[frame_start_index:frame_end_index] = np.sign(np.subtract(zcr[frame_start_index:frame_end_index],threshold*ave_zcr)+1)
        hzcrr[frame_start_index:frame_end_index] = np.subtract(zcr[frame_start_index:frame_end_index],threshold*ave_zcr)
        frame_start_index = frame_end_index
        ave_idx = ave_idx +1
    return hzcrr

def PCEN_MFCC(y,sr,parameters):
    '''
    Calculate the
        MFCCs : Mel-frequency cepstral coefficients (MFCCs)
        PCENs : using PCEN() to replace the log amplitude (dB) scaling on Mel spectra
    '''
    win_size = parameters['win_size']
    hop_size = parameters['hop_size']
    n_mels = parameters['num_mel_filters']
    n_dct = parameters['n_dct']
    fmin = parameters['min_freq']
    fmax = parameters['max_freq']

    S_MFCC = librosa.power_to_db(librosa.feature.melspectrogram(y=y, 
                                                                sr=sr,
                                                                n_fft=win_size, 
                                                                hop_length=hop_size, 
                                                                power=2, 
                                                                n_mels= n_mels,
                                                                fmin = fmin,
                                                                fmax = fmax))
    S_PCEN = librosa.pcen(librosa.feature.melspectrogram(y=y, 
                                                        sr=sr,
                                                        n_fft=win_size, 
                                                        hop_length=hop_size, 
                                                        power=1, 
                                                        n_mels= n_mels,
                                                        fmin = fmin,
                                                        fmax = fmax))

    return scipy.fftpack.dct(S_MFCC, axis=0, type=2, norm='ortho')[:n_dct],scipy.fftpack.dct(S_PCEN, axis=0, type=2, norm='ortho')[:n_dct]

def Spectral_Entropy(y_Mag,sr,n_short_blocks=10):
    Entropy = []
    eps = 0.00000001
    for i in range(y_Mag.shape[1]):
        X = y_Mag[:,i]
        L = len(X)                         
        Eol = np.sum(X ** 2)            #Spectral Energy in each frequency bin

        sub_win_len = int(np.floor(L / n_short_blocks))   
        if L != sub_win_len * n_short_blocks:
            X = X[0:sub_win_len * n_short_blocks]

        sub_wins = X.reshape(sub_win_len, n_short_blocks, order='F').copy()  
        s = np.sum(sub_wins ** 2, axis=0) / (Eol + eps)                      
        En = -np.sum(s*np.log2(s + eps))   
        Entropy.append(En)
    return Entropy
#=================================================================================================================

#======================================Testing Audio Processor=========================================
def read_audio(test_file,params):
    
    #================================Load Audio(Dowmsampled&Mono)=========================
    signal,sampling_rate = librosa.load(test_file, mono = True)
        
    #================================STFT(Framing&windowing)==============================
    signal_mag = np.abs(librosa.stft(signal, n_fft=params['win_size'], hop_length=params['hop_size']))       
    signal_pow = librosa.amplitude_to_db(signal_mag, ref=np.max)
    return signal,signal_mag,signal_pow,sampling_rate



