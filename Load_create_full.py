# -*- coding: utf-8 -*-
#"""
#Created on Thu Mar 26 10:37:40 2020
#
#@author: cyberguerra
#"""

import pandas as pd
import os
import shutil
import librosa
import librosa.display
import numpy as np
import pathlib
import csv
import matplotlib.pyplot as plt
from PIL import Image
import io
import sys
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
np.set_printoptions(threshold=sys.maxsize)
plt.ioff()

def Load_Create_Full(ref_csv, audio_folder):
##### Copy and organize wav files into categorical folders and subfolders #####
    ref_csv = ref_csv
    audio_folder = audio_folder
    cmap = plt.get_cmap('binary')

    # Load CSV for reference (filename,category,target and fold)
    data = pd.read_csv(ref_csv)
    
    colnames = list(data.iloc[:, 1:].columns)

    destination = r'AudioClassification/audio/'
    # Iterate through category names to derive subcategories and create second level of directory
    for row in data.itertuples():
        filename = row.filename
        for col in colnames:
            new_filename = os.path.join(destination, col, row._asdict()[col])
            old_filename = os.path.join(audio_folder, filename)
            if not os.path.exists(new_filename):
                os.makedirs(new_filename)
            shutil.copy(old_filename, new_filename)

##############################################################################
###############  CREATE SPECTROGRAPH REFERENCE CSV ###########################
# Preprocess
    # feature extraction and preprocessing data
    # Assign value to 'col' so we can use it to iterate through rows later
    for col in colnames:
        header = None
    def listToString(s):
        # initialize an empty string 
        str1 = " " 
        # return string   
        return (str1.join(s))
    column_names = listToString(colnames)
    
    header = 'filename {} chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate  '.format(column_names)
    
    # Extracting features from Spectrogram: Mel-frequency cepstral coefficients (MFCC)(20 in number), Spectral Centroid, Zero Crossing Rate, Chroma Frequencies, Spectral Roll-off.
    for i in range(1, 21):
        header += f' mfcc{i} '
    header = header.split()

    pathlib.Path(f'AudioClassification/audio_data/').mkdir(parents=True, 
                exist_ok=True)
    
    #Writing data to csv file
    file = open('AudioClassification/audio_data/audio_characteristics.csv', 
                'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)

    for row in data.itertuples():
        filename = row.filename
        audio_file = f'AudioClassification/audio/{col}/{row._asdict()[col]}/{filename}'
        y, sr = librosa.load(audio_file, 
                             mono=True, 
                             duration=30)
        rmse = librosa.feature.rms(y)
        chroma_stft = librosa.feature.chroma_stft(y=y, 
                                                  sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, 
                                                      sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, 
                                                     sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, 
                                                   sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, 
                                    sr=sr)
        to_append = f'{filename} '
        for col in colnames:
            to_append += f' {row._asdict()[col]} '
        to_append = to_append + f' {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)} '
        for e in mfcc:
            to_append += f' {np.mean(e)} '
        # Adds audio_characteristics.csv to DTG folder
        file = open('AudioClassification/audio_data/audio_characteristics.csv', 
                    'a', 
                    newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())

##############################################################################
################  CREATE SPECTOGRAM ARRAY #####################
# Preprocess
    SpecArrayImages = np.empty(shape=([72,72,0]))#,dtype='object') 
    # Sets color palette for audio image
    cmap = plt.get_cmap('binary')
    header = 'filename {} '.format(column_names)
    # Extracting features from Spectrogram: Mel-frequency cepstral coefficients (MFCC)(20 in number), Spectral Centroid, Zero Crossing Rate, Chroma Frequencies, Spectral Roll-off.
    for i in range(1, 5185):
        header += f' {i} '
    header = header.split()
########## TRAIN SPECTOGRAM ARRAY #############
    pathlib.Path(f'AudioClassification/image_data/').mkdir(parents=True, 
                exist_ok=True)
    #Writing data to csv file
    file = open('AudioClassification/image_data/spectogram_characteristics.csv', 
                'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    
    for row in data.itertuples():
        filename = row.filename
        audio_file = f'{audio_folder}/{filename}'
        y, sr = librosa.load(audio_file, 
                             mono=True, 
                             duration=30)
        fig=plt.figure(figsize=(1, 1))
        plt.specgram(y, NFFT=2048, 
                     Fs=2, 
                     Fc=0, 
                     noverlap=128, 
                     cmap=cmap, 
                     sides='default', 
                     mode='default', 
                     scale='dB');
        plt.axis('off')
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
    
        with io.BytesIO() as out:
            fig.savefig(out, format="png", dpi=72)  # Add dpi= to match your figsize
            plt.margins(0,0);
            pic = Image.open(out)

            arr = np.array(pic)
            
            # make a 1-dimensional view of arr
            flat_arr = arr[:,:,0].ravel().flatten()
            
            # convert it to a matrix
            vector = np.matrix(flat_arr)
            varray = np.array(vector).flatten()
            varray = str(varray).lstrip('[').rstrip(']')

            arr0 = arr[:,:,0] #Only use first matrix
            SpecArrayImages = np.dstack((SpecArrayImages, arr0))#.shape
            #print(*varray, sep =' ')
            to_append = f'{filename} '
            for col in colnames:
                to_append += f' {row._asdict()[col]} '
            to_append = to_append + f'{varray}'
        
            file = open('AudioClassification/image_data/spectogram_characteristics.csv', 
                        'a', 
                        newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())
        plt.close()

##############################################################################        
################  CREATE WAVEFORM ARRAY #####################
# Preprocess
    #Convert audio to image
    images = np.empty(shape=([72,72,0]))#,dtype='object') 
    # Sets color palette for audio image
    cmap = plt.get_cmap('binary')
    header = 'filename {} '.format(column_names)
    # Extracting features from Spectrogram: Mel-frequency cepstral coefficients (MFCC)(20 in number), Spectral Centroid, Zero Crossing Rate, Chroma Frequencies, Spectral Roll-off.
    for i in range(1, 5185):
        header += f' {i} '
    header = header.split()

########## TRAIN WAVEFORM ARRAY #############
    pathlib.Path(f'AudioClassification/image_data/').mkdir(parents=True, 
                exist_ok=True)
    #Writing data to csv file
    file = open('AudioClassification/image_data/waveform_characteristics.csv', 
                'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
        
    for row in data.itertuples():
        filename = row.filename
        audio_file = f'{audio_folder}/{filename}'
        wav, sr = librosa.load(audio_file, 
                             mono=True, 
                             duration=5)
        librosa.display.waveplot(wav, sr)
        plt.axis('off')
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
    
        with io.BytesIO() as out:
            fig.savefig(out, format="png", dpi=72)  # Add dpi= to match your figsize
            plt.margins(0,0);
            pic = Image.open(out)

            arr = np.array(pic)

            # make a 1-dimensional view of arr
            flat_arr = arr[:,:,0].ravel().flatten()
            
            # convert it to a matrix
            vector = np.matrix(flat_arr)
            varray = np.array(vector).flatten()
            varray = str(varray).lstrip('[').rstrip(']')
            arr0 = arr[:,:,0] #Only use first matrix
            images = np.dstack((images, arr0))#.shape
            to_append = f'{filename} '
            for col in colnames:
                to_append += f' {row._asdict()[col]} '
            to_append = to_append + f'{varray}'
        
            file = open('AudioClassification/image_data/waveform_characteristics.csv', 
                        'a', 
                        newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())
        plt.close()
            
################  CREATE SPECTOGRAM AND WAVEFORM IMAGES  #####################
########## SPECTOGRAM IMAGES #############
    # Iterate through category names to derive subcategories and create second level of directory
    for audio_name in os.listdir(audio_folder):
        audio_file = f'{audio_folder}/{audio_name}'
        y, sr = librosa.load(audio_file, 
                             mono=True, 
                             duration=30)
        plt.figure(figsize=(1, 1))
        plt.specgram(y, NFFT=2048, 
                     Fs=2, 
                     Fc=0, 
                     noverlap=128, 
                     cmap=cmap, 
                     sides='default', 
                     mode='default', 
                     scale='dB');
        plt.axis('off')
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
    
        for row in data.itertuples():
            filename = row.filename
            if audio_name == filename:
                for col in colnames:
                    pathlib.Path(f'AudioClassification/spectogram/{col}/{row._asdict()[col]}/').mkdir(parents=True, exist_ok=True)
                    plt.savefig(f'AudioClassification/spectogram/{col}/{row._asdict()[col]}/{filename[:-3].replace(".", "")}.png')


########## WAVEFORM IMAGES #############
    for audio_name in os.listdir(audio_folder):
        audio_file = f'{audio_folder}/{audio_name}'
        plt.figure(figsize=(1, 1))
        wav, sr = librosa.load(audio_file, 
                             mono=True, 
                             duration=5)
        librosa.display.waveplot(wav, sr)
        plt.axis('off');
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
    
        for row in data.itertuples():
            filename = row.filename
            if audio_name == filename:
                for col in colnames:
                    pathlib.Path(f'AudioClassification/waveform/{col}/{row._asdict()[col]}/').mkdir(parents=True, 
                                exist_ok=True)
                    plt.savefig(f'AudioClassification/waveform/{col}/{row._asdict()[col]}/{filename[:-3].replace(".", "")}.png')
                    
                
########## CREATE TRAIN/TEST SET (70/30) ###############################
    train, test = train_test_split(data, test_size=0.3, random_state=42, shuffle=True)
    
    traindestination = f'AudioClassification/Train/audio/'
    # Iterate through category names to derive subcategories and create second level of directory
    for row in train.itertuples():
        filename = row.filename
        for col in colnames:
            new_filename = os.path.join(traindestination, col, row._asdict()[col])
            old_filename = os.path.join(audio_folder, filename)
            if not os.path.exists(new_filename):
                os.makedirs(new_filename)
            shutil.copy(old_filename, new_filename)
                    
    testdestination = f'AudioClassification/Test/audio/'
    # Iterate through category names to derive subcategories and create second level of directory
    for row in test.itertuples():
        filename = row.filename
        for col in colnames:
            new_filename = os.path.join(testdestination, col, row._asdict()[col])
            old_filename = os.path.join(audio_folder, filename)
            if not os.path.exists(new_filename):
                os.makedirs(new_filename)
            shutil.copy(old_filename, new_filename)       
                    
    ###############  CREATE SPECTROGRAPH  ##############################
    #########  TRAIN  ########
    # feature extraction and preprocessing data
    
    # Assign value to 'col' so we can use it to iterate through rows later
    for col in colnames:
        header = None
    def listToString(s):
        # initialize an empty string 
        str1 = " " 
        # return string   
        return (str1.join(s))
    column_names = listToString(colnames)
    
    header = 'filename {} chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate  '.format(column_names)
    
    # Extracting features from Spectrogram: Mel-frequency cepstral coefficients (MFCC)(20 in number), Spectral Centroid, Zero Crossing Rate, Chroma Frequencies, Spectral Roll-off.
    for i in range(1, 21):
        header += f' mfcc{i} '
    #    for col in colnames:
    #        header += col+' '
    header = header.split()
    
    pathlib.Path(f'AudioClassification/Train/audio_data/').mkdir(parents=True, 
                exist_ok=True)
    
    #Writing data to csv file
    file = open('AudioClassification/Train/audio_data/audio_characteristicsTRAIN.csv', 
                'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
        
    for row in train.itertuples():
        filename = row.filename
        audio_file = f'AudioClassification/Train/audio/{col}/{row._asdict()[col]}/{filename}'
        y, sr = librosa.load(audio_file, 
                             mono=True, 
                             duration=30)
        rmse = librosa.feature.rms(y)
        chroma_stft = librosa.feature.chroma_stft(y=y, 
                                                  sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, 
                                                      sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, 
                                                     sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, 
                                                   sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, 
                                    sr=sr)
        to_append = f'{filename} '
        for col in colnames:
            to_append += f' {row._asdict()[col]} '
        to_append = to_append + f' {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)} '
        for e in mfcc:
            to_append += f' {np.mean(e)} '
        # Adds audio_characteristics.csv to DTG folder
        file = open('AudioClassification/Train/audio_data/audio_characteristicsTRAIN.csv', 
                    'a', 
                    newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())
            
    #########  TEST  ########
    pathlib.Path(f'AudioClassification/Test/audio_data/').mkdir(parents=True, 
                exist_ok=True)
    
    #Writing data to csv file
    file = open(f'AudioClassification/Test/audio_data/audio_characteristicsTEST.csv', 
                'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
        
    for row in test.itertuples():
        filename = row.filename
        audio_file = f'{audio_folder}/{filename}'
        y, sr = librosa.load(audio_file, 
                             mono=True, 
                             duration=30)
        rmse = librosa.feature.rms(y)
        chroma_stft = librosa.feature.chroma_stft(y=y, 
                                                  sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, 
                                                      sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, 
                                                     sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, 
                                                   sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, 
                                    sr=sr)
        to_append = f'{filename} '
        for col in colnames:
            to_append += f' {row._asdict()[col]} '
        to_append = to_append + f' {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)} '
        for e in mfcc:
            to_append += f' {np.mean(e)} '
        # Adds audio_characteristics.csv to DTG folder
        file = open(f'AudioClassification/Test/audio_data/audio_characteristicsTEST.csv', 
                    'a', 
                    newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())
            
    ################  CREATE IMAGES  #####################
    #### CREATE SPECTOGRAM    
    ########## TRAIN #############
    # Iterate through category names to derive subcategories and create second level of directory
    for audio_name in os.listdir(audio_folder):
        audio_file = f'{audio_folder}/{audio_name}'
        y, sr = librosa.load(audio_file, 
                             mono=True, 
                             duration=30)
        plt.figure(figsize=(1, 1))
        plt.specgram(y, NFFT=2048, 
                     Fs=2, 
                     Fc=0, 
                     noverlap=128, 
                     cmap=cmap, 
                     sides='default', 
                     mode='default', 
                     scale='dB');
        plt.axis('off')
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
    
        for row in train.itertuples():
            filename = row.filename
            if audio_name == filename:
                for col in colnames:
                    pathlib.Path(f'AudioClassification/Train/spectogram/{col}/{row._asdict()[col]}/').mkdir(parents=True, 
                                exist_ok=True)
                    plt.savefig(f'AudioClassification/Train/spectogram/{col}/{row._asdict()[col]}/{filename[:-3].replace(".", "")}.png')

    ######## TEST ###########
    # Iterate through category names to derive subcategories and create second level of directory
        for row in test.itertuples():
            filename = row.filename
            if audio_name == filename:
                for col in colnames:
                    pathlib.Path(f'AudioClassification/Test/spectogram/{col}/{row._asdict()[col]}/').mkdir(parents=True, 
                                exist_ok=True)
                    plt.savefig(f'AudioClassification/Test/spectogram/{col}/{row._asdict()[col]}/{filename[:-3].replace(".", "")}.png')

#### CREATE WAVEFORM ####
## TRAIN ##
    for audio_name in os.listdir(audio_folder):
        audio_file = f'{audio_folder}/{audio_name}'
        plt.figure(figsize=(1, 1))
        wav, sr = librosa.load(audio_file, 
                             mono=True, 
                             duration=5)
        librosa.display.waveplot(wav, sr)
        plt.axis('off');
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
    
        for row in train.itertuples():
            filename = row.filename
            if audio_name == filename:
                for col in colnames:
                    pathlib.Path(f'AudioClassification/Train/waveform/{col}/{row._asdict()[col]}/').mkdir(parents=True, 
                                exist_ok=True)
                    plt.savefig(f'AudioClassification/Train/waveform/{col}/{row._asdict()[col]}/{filename[:-3].replace(".", "")}.png')
## TEST ##
        for row in test.itertuples():
            filename = row.filename
            if audio_name == filename:
                for col in colnames:
                    pathlib.Path(f'AudioClassification/Test/waveform/{col}/{row._asdict()[col]}/').mkdir(parents=True, 
                                exist_ok=True)
                    plt.savefig(f'AudioClassification/Test/waveform/{col}/{row._asdict()[col]}/{filename[:-3].replace(".", "")}.png')
               
    ################  CREATE IMAGE ARRAYS #####################
    ### CREATE SPECTOGRAM ARRAY ##############
    ########## TRAIN #############
    #Convert audio to image
    images = np.empty(shape=([72,72,0]))#,dtype='object') 
    
    # Sets color palette for audio image
    cmap = plt.get_cmap('binary')
    
    header = 'filename {} '.format(column_names)
    
    # Extracting features from Spectrogram: Mel-frequency cepstral coefficients (MFCC)(20 in number), Spectral Centroid, Zero Crossing Rate, Chroma Frequencies, Spectral Roll-off.
    for i in range(1, 5185):
        header += f' {i} '
    #    for col in colnames:
    #        header += col+' '
    header = header.split()
    
    pathlib.Path(f'AudioClassification/Train/image_data/').mkdir(parents=True, 
                exist_ok=True)
    
    #Writing data to csv file
    file = open('AudioClassification/Train/image_data/image_characteristicsTRAIN.csv', 
                'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    
    for row in train.itertuples():
        filename = row.filename
        audio_file = f'{audio_folder}/{filename}'
        y, sr = librosa.load(audio_file, 
                             mono=True, 
                             duration=30)
        fig=plt.figure(figsize=(1, 1))
        plt.specgram(y, NFFT=2048, 
                     Fs=2, 
                     Fc=0, 
                     noverlap=128, 
                     cmap=cmap, 
                     sides='default', 
                     mode='default', 
                     scale='dB');
        plt.axis('off')
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
    
        with io.BytesIO() as out:
            fig.savefig(out, format="png", dpi=72)  # Add dpi= to match your figsize
            plt.margins(0,0);
            pic = Image.open(out)

            arr = np.array(pic)
            
            # make a 1-dimensional view of arr
            flat_arr = arr[:,:,0].ravel().flatten()
            
            # convert it to a matrix
            vector = np.matrix(flat_arr)
            varray = np.array(vector).flatten()
            varray = str(varray).lstrip('[').rstrip(']')
            arr0 = arr[:,:,0] #Only use first matrix
            images = np.dstack((images, arr0))#.shape
            #print(*varray, sep =' ')
            to_append = f'{filename} '
            for col in colnames:
                to_append += f' {row._asdict()[col]} '
            #to_append = to_append + f' {vector} '
            to_append = to_append + f'{varray}'
        
            file = open('AudioClassification/Train/image_data/image_characteristicsTRAIN.csv', 
                        'a', 
                        newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())
        plt.close()
        
    ########## TEST #############
    images = np.empty(shape=([72,72,0]))#,dtype='object') 
    
    # Sets color palette for audio image
    cmap = plt.get_cmap('binary')
    
    header = 'filename {} '.format(column_names)
    
    # Extracting features from Spectrogram: Mel-frequency cepstral coefficients (MFCC)(20 in number), Spectral Centroid, Zero Crossing Rate, Chroma Frequencies, Spectral Roll-off.
    for i in range(1, 5185):
        header += f' {i} '
    header = header.split()
    
    pathlib.Path(f'AudioClassification/Test/image_data/').mkdir(parents=True, 
                exist_ok=True)
    
    #Writing data to csv file
    file = open('AudioClassification/Test/image_data/image_characteristicsTEST.csv', 
                'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    
    for row in test.itertuples():
        filename = row.filename
        audio_file = f'{audio_folder}/{filename}'
        y, sr = librosa.load(audio_file, 
                             mono=True, 
                             duration=30)
        fig=plt.figure(figsize=(1, 1))
        plt.specgram(y, NFFT=2048, 
                     Fs=2, 
                     Fc=0, 
                     noverlap=128, 
                     cmap=cmap, 
                     sides='default', 
                     mode='default', 
                     scale='dB');
        plt.axis('off')
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
    
        with io.BytesIO() as out:
            fig.savefig(out, format="png", dpi=72)  # Add dpi= to match your figsize
            plt.margins(0,0);
            pic = Image.open(out)

            arr = np.array(pic)
            
            # make a 1-dimensional view of arr
            flat_arr = arr[:,:,0].ravel().flatten()
            
            # convert it to a matrix
            vector = np.matrix(flat_arr)
            varray = np.array(vector).flatten()
            varray = str(varray).lstrip('[').rstrip(']')
            arr0 = arr[:,:,0] #Only use first matrix
            images = np.dstack((images, arr0))#.shape
            #print(*varray, sep =' ')
            to_append = f'{filename} '
            for col in colnames:
                to_append += f' {row._asdict()[col]} '
            #to_append = to_append + f' {vector} '
            to_append = to_append + f'{varray}'
        
            file = open('AudioClassification/Test/image_data/image_characteristicsTEST.csv', 
                        'a', 
                        newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())
        plt.close()
        
    ### CREATE WAVEFORM ARRAY ##############
    ########## TRAIN #############
    #Convert audio to image
    images = np.empty(shape=([72,72,0]))#,dtype='object') 
    
    # Sets color palette for audio image
    cmap = plt.get_cmap('binary')
    
    header = 'filename {} '.format(column_names)
    
    # Extracting features from Spectrogram: Mel-frequency cepstral coefficients (MFCC)(20 in number), Spectral Centroid, Zero Crossing Rate, Chroma Frequencies, Spectral Roll-off.
    for i in range(1, 5185):
        header += f' {i} '
    #    for col in colnames:
    #        header += col+' '
    header = header.split()
    
    pathlib.Path(f'AudioClassification/Train/image_data/').mkdir(parents=True, 
                exist_ok=True)
    
    #Writing data to csv file
    file = open('AudioClassification/Train/image_data/waveform_characteristicsTRAIN.csv', 
                'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    
    for row in train.itertuples():
        filename = row.filename
        audio_file = f'{audio_folder}/{filename}'
        plt.figure(figsize=(1, 1))
        wav, sr = librosa.load(audio_file, 
                             mono=True, 
                             duration=5)
        librosa.display.waveplot(wav, sr)
        plt.axis('off');
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
    
        with io.BytesIO() as out:
            fig.savefig(out, format="png", dpi=72)  # Add dpi= to match your figsize
            plt.margins(0,0);
            pic = Image.open(out)

            arr = np.array(pic)
            
            # make a 1-dimensional view of arr
            flat_arr = arr[:,:,0].ravel().flatten()
            
            # convert it to a matrix
            vector = np.matrix(flat_arr)
            varray = np.array(vector).flatten()
            varray = str(varray).lstrip('[').rstrip(']')
            arr0 = arr[:,:,0] #Only use first matrix
            images = np.dstack((images, arr0))#.shape
            #print(*varray, sep =' ')
            to_append = f'{filename} '
            for col in colnames:
                to_append += f' {row._asdict()[col]} '
            #to_append = to_append + f' {vector} '
            to_append = to_append + f'{varray}'
        
            file = open('AudioClassification/Train/image_data/waveform_characteristicsTRAIN.csv', 
                        'a', 
                        newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())
        plt.close()
        
    ########## TEST #############
    #Convert audio to image
    images = np.empty(shape=([72,72,0]))#,dtype='object') 
    
    # Sets color palette for audio image
    cmap = plt.get_cmap('binary')
    
    header = 'filename {} '.format(column_names)
    
    # Extracting features from Spectrogram: Mel-frequency cepstral coefficients (MFCC)(20 in number), Spectral Centroid, Zero Crossing Rate, Chroma Frequencies, Spectral Roll-off.
    for i in range(1, 5185):
        header += f' {i} '
    #    for col in colnames:
    #        header += col+' '
    header = header.split()
    
    pathlib.Path(f'AudioClassification/Test/image_data/').mkdir(parents=True, 
                exist_ok=True)
    
    #Writing data to csv file
    file = open('AudioClassification/Test/image_data/waveform_characteristicsTEST.csv', 
                'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    
    for row in test.itertuples():
        filename = row.filename
        audio_file = f'{audio_folder}/{filename}'
        plt.figure(figsize=(1, 1))
        wav, sr = librosa.load(audio_file, 
                             mono=True, 
                             duration=5)
        librosa.display.waveplot(wav, sr)
        plt.axis('off');
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
    
        with io.BytesIO() as out:
            fig.savefig(out, format="png", dpi=72)  # Add dpi= to match your figsize
            plt.margins(0,0);
            pic = Image.open(out)

            arr = np.array(pic)
            
            # make a 1-dimensional view of arr
            flat_arr = arr[:,:,0].ravel().flatten()
            
            # convert it to a matrix
            vector = np.matrix(flat_arr)
            varray = np.array(vector).flatten()
            varray = str(varray).lstrip('[').rstrip(']')

            arr0 = arr[:,:,0] #Only use first matrix
            images = np.dstack((images, arr0))#.shape
            to_append = f'{filename} '
            for col in colnames:
                to_append += f' {row._asdict()[col]} '
            to_append = to_append + f'{varray}'
        
            file = open('AudioClassification/Test/image_data/waveform_characteristicsTEST.csv', 
                        'a', 
                        newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())
        plt.close()

Load_Create_Full('GameAll.csv', 'VGAudio')
