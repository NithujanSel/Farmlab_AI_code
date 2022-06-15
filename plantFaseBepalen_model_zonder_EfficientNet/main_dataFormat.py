import numpy as np
import pandas as pd
from pathlib import Path
import os
import random
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle

DIRECTORY = r"C:\Users\Nithusan Kanees\Desktop\AP Hogeschool\Jaar2\S2\21-22 IoT Project\AI\ModelBladGrotte\train"
CATEGORIES = ["fase1","fase2","fase3","fase4"]
IMG_SIZE = 150
data = []

for categorie in CATEGORIES:
    folder = os.path.join(DIRECTORY,categorie)
    label = CATEGORIES.index(categorie)
    for img in os.listdir(folder):
        img_path = os.path.join(folder,img)
        img_array = cv2.imread(img_path)
        img_array = cv2.resize(img_array,(IMG_SIZE ,IMG_SIZE ))
        # plt.imshow(img_array)
        # plt.show()
        # break
        data.append([img_array,label])


random.shuffle(data)

from sklearn.model_selection import train_test_split
training_data, testing_data = train_test_split(data, test_size=0.2, random_state=25)

X_t = []
Y_t = []

X_te = []
Y_te = []

for feature,label in training_data:
    X_t.append(feature)
    Y_t.append(label)

for feature,label in testing_data:
    X_te.append(feature)
    Y_te.append(label)


X_train = np.array(X_t)
Y_train = np.array(Y_t)

X_test= np.array(X_te)
Y_test = np.array(Y_te)

pickle.dump(X_train,open("X_train.pkl","wb"))
pickle.dump(Y_train,open("Y_train.pkl","wb"))

pickle.dump(X_test,open("X_test.pkl","wb"))
pickle.dump(Y_test,open("Y_test.pkl","wb"))