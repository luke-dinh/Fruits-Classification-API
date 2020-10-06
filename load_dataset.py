import numpy as np
import os
import pickle
import cv2 
import random

datadir = "Datasets"

categories = ["Apple", "Lemon", "Mango", "Raspberry"]

training_data = []

for category in categories:
    path = os.path.join(datadir, category)
    class_label = categories.index(category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        training_data.append([img_array, class_label])

random.shuffle(training_data)

#check whether the data was shuffled corectly or not
for sample in training_data[:10]:
    print(sample[1])

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, 100, 100, 1)

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)
#print(X[1]) // Check 

