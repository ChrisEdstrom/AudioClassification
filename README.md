# AudioClassification
Requirements:
- A folder with named wav files
- a CSV with named wav files as a column (preferably the first column) and category columns

 Load_Create:
 - Copies wav files referenced in the ref_csv into named folders by category
 - Creates an audio spectograph of 26-features
 - Creates an image spectograph of 5184 features based on pixel color
 - Creates spectogram images and copies them to folders based on category
 - Creates waveform images and copies them to folders based on category
 - Splits the data set 70/30 and creates all the aforementioned for each. 
 
 The GUI compares audio files by audio features, image features and simple image comparison.
 "Select Audio File" is the file you want to compare. It can be either .wav for comparing audio features or png for Image Feature and Image comparison.
 "Select Reference CSV" is either the audio or image feature CSV created during Load_Create and is used for "Compare Audio Feature" and "Compare Image Feature"
 "Select Reference Folder" is used for "Compare Image"
 "Show Image" displayes a small spectogram of the "Select Audio File"
 
 
 
