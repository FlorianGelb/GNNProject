from  sklearn.svm import SVC
from DataPreparation.CustomDataSet import CustomDataSet
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision import datasets

def transform(input):
    input = torch.FloatTensor(np.array(input))
    input = input.flatten()
    input = input.type(torch.FloatTensor)
    input -= torch.min(input)
    input /= torch.max(input)
    return input

def return_sklearn_data_set(size):
    print()
    data_set = datasets.FashionMNIST("/FashionMNIST/", download=False, transform=transform)
    data_loader = DataLoader(data_set, batch_size=size)
    X =  next(iter(data_loader))[0].numpy().reshape(size, -1)
    y = next(iter(data_loader))[1].numpy()
    return X, y
X, y = return_sklearn_data_set(60000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)


svc_model = SVC(C=10, kernel="rbf").fit(X_train, y_train)
y_pred = svc_model.predict(X_test)
c_matrix = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(c_matrix).plot()
plt.show()
print(svc_model.score(X_test, y_test))





with open("FashionMNISTClassifier.pkl", "wb+") as file:
    pickle.dump(svc_model, file)
