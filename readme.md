# Sources:
## Data Corruption (DataPreparation/Corruption): https://github.com/google-research/mnist-c
## Sparse Autoencoder (K-Sparse): https://github.com/AntonP999/Sparse_autoencoder
## LAP: https://github.com/alinlab/lookahead_pruning



# Generate Corrupted datasets:
Execute DataPreparation/CorruptFashionMNIST.py 

Execute DataPreparation/CorruptMNIST.py

Executing those will result in the creation of the folders CorruptedFashionMNIST and CorruptedMNIST.
Those will contain the actual images and a csv file mapping the images to their lables.
To use the corrupted data, CustomDataSet is needed (based on https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).
Its parameters are the path to the folder containing the images and the path to the csv file.

# Classifiers:
Classifiers can be trained using the scripts in Classifiers/ or by downloading them from google drive.
The links are stored in Classifiers/classifier_downloads.
They are stored using pickle.
