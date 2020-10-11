# AudioClassification
Load_Create Requirements:
- A folder with named wav files
- a CSV with named wav files as a column (preferably the first column) and category columns

 Load_Create Functions:
 - Copies wav files referenced in the ref_csv into named folders by category
 - Creates an audio spectograph of 26-features
 - Creates an image spectograph of 5184 features based on pixel color
 - Creates spectogram images and copies them to folders based on category
 - Creates waveform images and copies them to folders based on category
 - Splits the data set 70/30 and creates all the aforementioned for each. 
 
tKinter_Audio-Image_Feature_ComparisonGUI Requirements:
- A wav for audio feature comparison or png file for image comparison
- a feature CSV
  - An audio feature CSV; i.e. audio_characteristics.csv created from Load_Create, for audio feature comparison
  - An image feature CSV; i.e. spectogram_characteristics.csv created from Load_Create, for image feature comparison
- A folder of images; i.e. AudioClassification/spectogram

 The GUI compares audio files by audio features, image features and simple image comparison.
 - "Select Audio File" is the file you want to compare. It can be either .wav for comparing audio features or png for Image Feature and Image comparison.
 - "Select Reference CSV" is either the audio or image feature CSV created during Load_Create and is used for "Compare Audio Feature" and "Compare Image Feature"
 - "Select Reference Folder" is used for "Compare Image"
 - "Show Image" displayes a small spectogram of the "Select Audio File"
 - "Compare Audio Feature" extracts 26 audio features from the selected audio file and compares them for similarity against the selected reference CSV. It provides a predicted category and displays the top 5 probabilities.
 - "Compare Image Feature" flattens the imagae 72x72 pixel array of the selected audio file into a 1 by 5184 row and uses ExtraTrees to predict a similar wav or png compared to the spectograph feature CSV. Waveforms are not created yet.
 - "Compare Image" takes the png or converted wav and uses Root Square Means Difference (RMSD), Structural Similarity Method (SSIM), and Absolute Difference (DIFF) to compare images and provide predictions based on majority rules; for example, if RMSD and DIFF show the same prediction similarity, that prediction will be displayed. If no majority, then the prediction is unspecified.
 
 
 
