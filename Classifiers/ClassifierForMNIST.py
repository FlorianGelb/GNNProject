from  sklearn.svm import SVC
from DataPreparation.CorruptedFashionMNISTDataSet import CustomDataSet
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST

def return_sklearn_data_set(size):
    data_set = CustomDataSet("DataPreparation/CorruptedMNIST/Names.csv", "DataPreparation/CorruptedMNIST")
    data_loader = DataLoader(data_set, batch_size=size)
    X =  next(iter(data_loader))[0].numpy().reshape(size, -1)
    y = next(iter(data_loader))[1].numpy()
    return X, y
X, y = return_sklearn_data_set(60000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)
#HG = HalvingGridSearchCV(SVC(), param_grid={"C": [i for i in range(2, 14, 2)], "kernel": ["linear", "rbf"]}, cv=3, verbose=5).fit(X_train, y_train)
#print(HG.best_estimator_.score(X_test, y_test))
#print(HG.best_params_)

svc_model = SVC(C=100, kernel="poly").fit(X_train, y_train)
print(svc_model.score(X_test, y_test))
y_pred = svc_model.predict(X_test)
c_matrix = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(c_matrix).plot()
plt.show()




with open("MNISTClassifier.pkl", "wb+") as file:
    pickle.dump(svc_model, file)

