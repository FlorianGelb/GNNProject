from  sklearn.svm import SVC
from DataPreparation.CorruptedFashionMNISTDataSet import CorruptedMNISTDataSet
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def return_sklearn_data_set(size):
    data_set = CorruptedMNISTDataSet("DataPreparation/CorruptedData/Names.csv", "DataPreparation/CorruptedData")
    data_loader = DataLoader(data_set, batch_size=size)
    X =  next(iter(data_loader))[0].numpy().reshape(size, -1)
    y = next(iter(data_loader))[1].numpy()
    return X, y
X, y = return_sklearn_data_set(30000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)
print("start")
svc = SVC(C= 10, kernel="rbf")
print(svc.fit(X_train, y_train).score(X_test, y_test))