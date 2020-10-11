# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 15:54:03 2019

@author: Chris
"""
from tkinter import *
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import librosa
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from pathlib import Path  
import io
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageTk
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
import warnings
warnings.filterwarnings('ignore')
import sys
import os
from PIL import Image
import math, operator
import functools
from PIL import ImageChops
from skimage.measure import compare_ssim
import cv2
import csv
import pathlib


np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=2) 
cmap = plt.get_cmap('binary')

bg='#CBD1DF'
class App(tk.Frame):

    def __init__(self, master):
        self.master = master

#Label identifies what GUI does
        lbl1 = tk.Label(self.master, 
                        text = "Select Audio File and Reference CSV then click Compare",
                        font=('Arial',16,'bold'),bg=bg)
        lbl1.grid(row = 0, column = 0, columnspan = 20)

#Entry 1 (Audio File)
        self.entry1 = tk.Entry(self.master,width = 60)
        self.entry1.grid(row = 1, column = 2)

#Entry 2 (Ref_CSV)
        self.entry2 = tk.Entry(self.master, width = 60)
        self.entry2.grid(row = 2, column = 2)

#Entry 3 (known_folder)
        self.entry3 = tk.Entry(self.master, width = 60)
        self.entry3.grid(row = 3, column = 2)
        
##Entry 4 (test_folder)
#        self.entry4 = tk.Entry(self.master, width = 50)
#        self.entry4.grid(row = 4, column = 2)

#Button 1 (Audio_File-One to Many)
        btn1 = tk.Button(self.master, text = "Select Audio File", command = self.load_audio)
        btn1.grid(row = 1, column = 1)
#Button 2 (Ref_CSV-One to Many, Many to Many)
        btn2 = tk.Button(self.master, text = "Select Reference CSV", command = self.load_ref_csv)
        btn2.grid(row = 2, column = 1)
#Button 9 (known_folder-One to Many, Many to Many)
        btn9 = tk.Button(self.master, text = "Select Reference Folder", command = self.load_known_folder)
        btn9.grid(row = 3, column = 1)
##Button 10 (test_folder-Many to Many)
#        btn10 = tk.Button(self.master, text = "Select Comparison Folder", command = self.load_unknown_folder)
#        btn10.grid(row = 4, column = 1)

#Button 3 Compare Audio Features
        btn3 = tk.Button(self.master, text = "Compare Audio Feature", command = self.single_audio_feature_comparison)
        btn3.grid(row = 8, column = 1)
#Button 8 Compare Image Features
        btn8 = tk.Button(self.master, text = "Compare Image Feature", command = self.single_image_feature_comparison_Pruned)
        btn8.grid(row = 10, column = 1)
#Button 13 Compare Images
        btn13 = tk.Button(self.master, text = "Compare Image", command = self.single_image_comparison)
        btn13.grid(row = 12, column = 1)
        
#Button 4 Quit
        btn4 = tk.Button(self.master, text='Quit', command=master.destroy).grid(row=16, column=3,
                        sticky=W, pady=4)
#Button 5 Clear Entry 1 (Audio_File)
        btn5 = tk.Button(self.master, text='Clear', command=self.clear_audio_text).grid(row=1, column=3,
                        sticky=W, pady=4)
#Button 6 Clear Entry 2 (Ref_CSV)
        btn6 = tk.Button(self.master, text='Clear', command=self.clear_csv_text).grid(row=2, column=3,
                sticky=W, pady=4)
#Button 11 Clear Entry 3 (Known Folder)
        btn11 = tk.Button(self.master, text='Clear', command=self.clear_known_folder).grid(row=3, column=3,
                        sticky=W, pady=4)
#Button 7 Show Audio File Image
        btn7 = tk.Button(self.master, text='Show Image', command=self.show_image).grid(row=6, column=2,
        sticky=W, pady=4)


#Compare Audio Features field
        self.text1 = tk.Text(self.master, height = 6, width = 60, wrap = 'word',bg=bg)
        vertscroll = tk.Scrollbar(self.master)
        vertscroll.config(command=self.text1.yview)
        self.text1.config(yscrollcommand=vertscroll.set)
        self.text1.grid(row=8,column=2)
        vertscroll.grid(row=8,column=20, sticky='NS')
#Compare Image Features field
        self.text2 = tk.Text(self.master, height = 6, width = 60, wrap = 'word',bg=bg)
        vertscroll = tk.Scrollbar(self.master)
        vertscroll.config(command=self.text2.yview)
        self.text2.config(yscrollcommand=vertscroll.set)
        self.text2.grid(row=10,column=2)
        vertscroll.grid(row=10,column=20, sticky='NS')
#Compare Image field
        self.text3 = tk.Text(self.master, height = 6, width = 60, wrap = 'word',bg=bg)
        vertscroll = tk.Scrollbar(self.master)
        vertscroll.config(command=self.text3.yview)
        self.text3.config(yscrollcommand=vertscroll.set)
        self.text3.grid(row=12,column=2)
        vertscroll.grid(row=12,column=20, sticky='NS')
        
        self.file_name_str = tk.StringVar()

#Load audio file; WAV or PNG image
    def load_audio(self):
        Aud = filedialog.askopenfilename(filetypes=([('WAV files', '*.wav'),('PNG files', '*.png')]))
        self.entry1.delete(0, 'end')
        self.entry1.insert(1, Aud)
    def load_ref_csv(self):
        ref = filedialog.askopenfilename(filetypes=([('CSV files', '*.csv')]))
        self.entry2.delete(0, 'end')
        self.entry2.insert(1, ref)
    def load_known_folder(self):
        knownF = filedialog.askdirectory()
        self.entry3.delete(0, 'end')
        self.entry3.insert(1, knownF + '/')
    
    def clear_audio_text(self):
        self.entry1.delete(0, 'end')
    def clear_csv_text(self):
        self.entry2.delete(0, 'end')
    def clear_known_folder(self):
        self.entry3.delete(0, 'end')
        
    def show_image(self):
        audio_file = self.entry1.get()
        if audio_file.endswith('.wav'):
            y, sr = librosa.load(audio_file, 
                                 mono=True, 
                                 duration=5)
            fig=plt.figure(figsize=(1, 1))
            plt.specgram(y, NFFT=2048, 
                         Fs=2, 
                         Fc=0, 
                         noverlap=128, 
                         cmap='inferno', 
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
                photo = ImageTk.PhotoImage(pic)
                label = tk.Label(self.master, image = photo)
                label.image = photo
                label.grid(row = 6,column = 2)
                image_file = photo

            fig2=plt.figure(figsize=(6,3))

#            filename = test[test['category']=='Voice'].iloc[0]['filename']
            wav, sr = librosa.load(audio_file, sr=None)
            librosa.display.waveplot(wav, sr)
            plt.title(audio_file)
#            plt.savefig('WavForm.png')
            with io.BytesIO() as out:
                fig2.savefig(out, format="png", dpi=72)  # Add dpi= to match your figsize
                plt.margins(0,0);
#                wav, sr = librosa.load(audio_file, sr=None)
#                wave = librosa.display.waveplot(wav, sr)
                image_file = Image.open(out)
                photo = ImageTk.PhotoImage(image_file)
                label = tk.Label(self.master, image = photo)
                label.image = photo
                label.grid(row = 6,column = 3)
        
        else:
            image_file = Image.open(audio_file)
            photo = ImageTk.PhotoImage(image_file)
            label = tk.Label(self.master, image = photo)
            label.image = photo
            label.grid(row = 6,column = 2)
            


    
#        librosa.display.specshow(spec_to_image(get_melspectrogram_db(filename, sr)), 
#                                 cmap='viridis')
#        plt.title(filename)
#        plt.savefig('SpecForm.png')

    def single_audio_feature_comparison(self):
        self.text1.delete(1.0,'end') #Clears/resets audio feature results when executed
        audio_file = self.entry1.get()
        
        self.file_name_str.set(Path(audio_file).name) #Displays audio filename in GUI
        tk.Label(self.master, textvariable=self.file_name_str, bg=bg).grid(row = 5, 
                       column = 2)
        #Create spectogram characteristics for single WAV file
        y, sr = librosa.load(audio_file, 
                             mono=True, 
                             duration=5)#30
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
        AudChar = [np.mean(chroma_stft),np.mean(rmse),np.mean(spec_cent),
                   np.mean(spec_bw),np.mean(rolloff),np.mean(zcr)]  
        for e in mfcc:
            AudChar += [np.mean(e)]
        
        #Retrieve KNOWN spectograms and create data frame for comparison
        df = pd.read_csv(self.entry2.get())
        #df = df.drop(['filename','Activity'],axis=1)
        
        r = range(len(df.columns)-26)

        def convertTuple(tup): 
            str =  ''.join(tup) 
            return str

        for n in r:
            # Assign X and y
            X=df.iloc[:,(len(df.columns)-26):].values 
            y=df.iloc[:,n].values

            # Split-out validation dataset
            X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=42)
            
            clf = DecisionTreeClassifier(random_state=0)
            path = clf.cost_complexity_pruning_path(X_train, Y_train)
            ccp_alphas = path.ccp_alphas
            
            clfs = []
            for ccp_alpha in ccp_alphas:
                clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
                clf.fit(X_train, Y_train)
                clfs.append(clf)

            clfs = clfs[:-1]
            ccp_alphas = ccp_alphas[:-1]
            
            X = X_test
            y = Y_test
            y_set = sorted(set(y))
        
            # define the model
            model = ExtraTreesClassifier(n_estimators=100)
            datavector = []
            
            for i in range(100):
                # fit the model on the whole dataset
                model.fit(X, y)
                # make a single prediction
    
                yhat = model.predict([AudChar])
                ynew = model.predict_proba([AudChar])
                datavector.append(yhat[0])
    
            d = {}
            
            for i in datavector:
                dkeys = list(d.keys())
                if i in d:
                    d[i] +=1 
    
                else:
                    d[i] = 1
    
            list1 = ynew.tolist()
            list1 = np.transpose(list1)
            dkeys = sorted(set(dkeys))
    
            floatList = [float(x) for x in list1]
    
            res = {y_set[i]: floatList[i] for i in range(len(y_set))} 
            sort_res = sorted(res.items(), key=lambda x: x[1], reverse=True)
            sort_cat = sorted(d.items(), key=lambda x: x[1], reverse=True)

            self.text1.insert('end','Predicted {} Class: '.format(df.columns[n])+yhat[0]+'\n')
            for i in sort_cat:
                self.text1.insert('end',' ')
                self.text1.insert('end',round(i[1]))
                self.text1.insert('end','% ')
                self.text1.insert('end',i[0] +'\n')

            #Print top 3 results by highest probability
            self.text1.insert('end','Probabilities by {}:'.format(df.columns[n])+'\n')
            for i in sort_res[:3]:
                self.text1.insert('end',' ')
                self.text1.insert('end', i[1])
                self.text1.insert('end','% ')
                self.text1.insert('end',i[0] +'\n')

    def single_image_feature_comparison_Pruned(self):
        self.text2.delete(1.0,'end')
        audio_file = self.entry1.get()
        self.file_name_str.set(Path(audio_file).name)
        tk.Label(self.master, textvariable=self.file_name_str, bg=bg).grid(row = 5, 
                       column = 2)
        y, sr = librosa.load(audio_file, 
                             mono=True, 
                             duration=5)
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
            
            pix = np.array(pic.getdata(), 
                           dtype=np.uint8)
            arr = np.array(pic)
            
            # make a 1-dimensional view of arr
            flat_arr = arr[:,:,0].ravel().flatten()
            
            # convert it to a matrix
            vector = np.matrix(flat_arr)
            varray = np.array(vector).flatten()
            varray = str(varray).lstrip('[').rstrip(']')
            row_arrays = pd.DataFrame(vector)
        
        ImgChar = row_arrays
        #Retrieve KNOWN spectograms and create data frame for comparison
        df = pd.read_csv(self.entry2.get())
        #df = df.drop(['filename'],axis=1)
        
        r = range(len(df.columns)-5184)
        print(audio_file)
        
        def Convert(lst): 
            res_dct = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)} 
            return res_dct 
        
        def merge(list1, list2): 
              
            merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))] 
            return merged_list 
        
        for n in r:
            # Assign X and y
            X=df.iloc[:,(len(df.columns)-5184):].values 
            y=df.iloc[:,n].values
            # Split-out validation dataset
            X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=42)
            
            clf = DecisionTreeClassifier(random_state=0)
            path = clf.cost_complexity_pruning_path(X_train, Y_train)
            ccp_alphas = path.ccp_alphas
            
            clfs = []
            for ccp_alpha in ccp_alphas:
                clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
                clf.fit(X_train, Y_train)
                clfs.append(clf)
            
            clfs = clfs[:-1]
            ccp_alphas = ccp_alphas[:-1]
            
            X = X_test
            y = Y_test
            y_set = sorted(set(y))
        
            # define the model
            model = ExtraTreesClassifier(n_estimators=100)
            datavector = []
            
            for i in range(100):
                # fit the model on the whole dataset
                model.fit(X, y)
                # make a single prediction
        
                yhat = model.predict([ImgChar])
                ynew = model.predict_proba([ImgChar])
                datavector.append(yhat[0])
        
            d = {}
        
            for i in datavector:
                dkeys = list(d.keys())
                if i in d:
                    d[i] +=1 
        
                else:
                    d[i] = 1
        
            list1 = ynew.tolist()
            list1 = np.transpose(list1)
            dkeys = sorted(set(dkeys))
        
            floatList = [float(x) for x in list1]
        
            res = {y_set[i]: floatList[i] for i in range(len(y_set))} 
            sort_res = sorted(res.items(), key=lambda x: x[1], reverse=True)
            sort_cat = sorted(d.items(), key=lambda x: x[1], reverse=True)
            self.text2.insert('end','Predicted {} Class: '.format(df.columns[n])+yhat[0]+'\n')

    def single_image_comparison(self):
        known_folder = self.entry3.get()
        audio_file = self.entry1.get()
        self.text3.delete(1.0,'end')
        self.file_name_str.set(Path(audio_file).name)
        tk.Label(self.master, textvariable=self.file_name_str, bg=bg).grid(row = 5, 
                       column = 2)

        if audio_file.endswith('.wav'):
            y, sr = librosa.load(audio_file, 
                                 mono=True, 
                                 duration=5)
            fig=plt.figure(figsize=(1, 1))
            plt.specgram(y, NFFT=2048, 
                         Fs=2, 
                         Fc=0, 
                         noverlap=128, 
                         cmap='binary', 
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
                image_file = Image.open(out)
                arr = np.array(image_file)
                grayB = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) #Convert the images to grayscale

        else:
            image_file = Image.open(audio_file)
            im2 = Image.open(audio_file)
            imageB = cv2.imread(audio_file)
            grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)# Convert the images to grayscale
                    
        d_folder = known_folder
        r_folder = known_folder
        s_folder = known_folder
            
            # Walk through folders and subfolder
        def list_files(dir):
            r = []
            for root, dirs, files in os.walk(dir):
                for name in files:
                    r.append(os.path.join(name))
            return r
        # Identify subfolders
        def list_folders(dir):
            dirlist = [item for item in os.listdir(dir) if os.path.isdir(os.path.join(dir, item)) ]
            return dirlist
        
        def listToString(s):
                # initialize an empty string 
                str1 = " " 
                # return string   
                return (str1.join(s))
            
        # Get histogram for both images and then getting a RMS
        # Calculate the root-mean-square difference between two images
        def rmsdiff(im1, im2):
            h = ImageChops.difference(im1, im2).histogram()
            # calculate rms
            return math.sqrt(functools.reduce(operator.add,
                map(lambda h, i: h*(i**2), h, range(256))
            ) / (float(im1.size[0]) * im1.size[1]))
        
        d=0
        s=0
        r=0
        img2 = image_file
        for folders in list_folders(known_folder):
            new_folder = known_folder+folders
            im2 = img2
            min_rmsd = 9999
            best_rmsd = None
            max_ssim = 0
            best_ssim = None
            min_differ = 9999
            best_differ = None
            for subdir, dirs, files in os.walk(new_folder):
                #print(subdir)
                for img1 in files:
                    im1 = Image.open(f'{subdir}/{img1}')
                    imageA = cv2.imread(f'{subdir}/{img1}')
                    #print(img1)
                    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
                    #RMSD
                    rmsd = rmsdiff(im2, im1)
                    #SSIM
                    (ssim, diff) = compare_ssim(grayA, grayB, full=True)
                    diff = (diff * 255).astype("uint8")
                    #DIFFERENCE
                    pairs = zip(im1.getdata(), im2.getdata())
                    if len(im1.getbands()) == 1:
                        # for gray-scale jpegs
                        dif = sum(abs(p1-p2) for p1,p2 in pairs)
                    else:
                        dif = sum(abs(c1-c2) for p1,p2 in pairs for c1,c2 in zip(p1,p2))
                    ncomponents = im1.size[0] * im1.size[1] * 3
                    differ = (dif / 255.0 * 100) / ncomponents
                    if rmsd < min_rmsd:# and rmsd > 0:
                        min_rmsd = rmsd
                        best_rmsd = img1
                        r_folder = subdir
                    if ssim > max_ssim:# and ssim < 1:
                        max_ssim = ssim
                        best_ssim = img1
                        s_folder = subdir
                    if differ < min_differ:# and differ > 0:
                        min_differ = differ
                        best_differ = img1
                        d_folder = subdir
                if d_folder is r_folder or d_folder is s_folder:
                    rec_folder = folders, os.path.basename(os.path.normpath(d_folder))
                    d+=1
                if s_folder is r_folder or s_folder is d_folder:
                    rec_folder = folders, os.path.basename(os.path.normpath(s_folder))
                    s+=1
                if r_folder is s_folder or r_folder is d_folder:
                    rec_folder = folders, os.path.basename(os.path.normpath(r_folder))
                    r+=1
        
                else:
                    rec_folder = "Unspecified"
            #self.text3.insert('end',*rec_folder,'\n')
            self.text3.insert('end',rec_folder)
            self.text3.insert('end','\n')


        self.text3.insert('end',f'RMSD: ')
        self.text3.insert('end',format(min_rmsd,'.0f') + ' ')
        self.text3.insert('end',best_rmsd +'\n')#,' - ',os.path.basename(os.path.normpath(r_folder)))

        self.text3.insert('end',f'SSIM: ')
        self.text3.insert('end','{:.0f}%'.format(max_ssim*100))
        self.text3.insert('end',' Similarity to ')
        self.text3.insert('end',best_ssim +'\n')#,' - ',os.path.basename(os.path.normpath(s_folder)))
        
        self.text3.insert('end',f'DIFF: ')
        self.text3.insert('end','{:.0f}%'.format(min_differ))
        self.text3.insert('end',' Difference from ')
        self.text3.insert('end',best_differ +'\n')#,' - ',os.path.basename(os.path.normpath(d_folder)))

if __name__ == "__main__":

    root = tk.Tk()
    myapp = App(root)
    root.configure(bg=bg)
    root.title("Feature Comparison")
    root.mainloop()