import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
import Corruption.corruptions as c
import csv
from PIL import Image
import os
import uuid
from tqdm import tqdm



data_set = datasets.FashionMNIST("/FashionMNIST/", download=True, transform=c.fog)
data_loader = DataLoader(data_set, batch_size=1)


label_name_dict = []

for image, label in tqdm(data_loader):
    image = Image.fromarray(image.detach().numpy()[0]).convert('L')
    filename = str(uuid.uuid4()) + ".png"
    label_name_dict.append({"Filename": filename, "Label": int(label)})
    image.save("CorruptedFashionMNIST"+os.sep+filename)


with open("CorruptedFashionMNIST" + os.sep + 'Names.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["Filename", "Label"])
    writer.writeheader()
    for data in label_name_dict:
        writer.writerow(data)