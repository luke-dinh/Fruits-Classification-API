import numpy as np
import skimage
import os
import pickle

dataset_dir = "Datasets"

fruits = ["Apple", "Lemon", "Mango", "Raspberry"]

data_features = np.zeros(shape= (1968, 360))
outputs = np.zeros(shape=(1968))

index = 0
label = 0

for fruit_dir in fruits:
    current_dir = os.path.join(dataset_dir, fruit_dir)   
    for img in os.listdir(current_dir):
        fruit_data = skimage.io.imread(fname = os.getcwd() + '/' + current_dir + '/' + img, as_gray=False)
        fruit_data_hsv = skimage.color.rgb2hsv(fruit_data)
        hist = np.histogram(fruit_data_hsv[:,:,0], bins= 360)
        data_features[index, :] = hist[0]
        outputs[index] = label
        index += 1
    label += 1

with open("data_features.pkl", "wb") as f:
    pickle.dump("data_features.pkl", f)

with open("outputs.pkl", "wb") as f:
    pickle.dump("outputs.pkl", f)





