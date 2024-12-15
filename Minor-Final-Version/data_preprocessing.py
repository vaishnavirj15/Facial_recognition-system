# data_preprocessing.py

import os
import cv2
import numpy as np
from random import shuffle
from tqdm import tqdm
from label import my_label

def my_data():
    """
    Loads and preprocesses the images from the 'data' folder.
    Converts them to grayscale and resizes to 50x50 pixels, 
    then associates them with the corresponding label.
    """
    data = []
    for img in tqdm(os.listdir("data")):  # Iterates through each image in 'data' folder
        path = os.path.join("data", img)  # Full path to image
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Reads the image as grayscale
        img_data = cv2.resize(img_data, (50, 50))  # Resize to 50x50 pixels
        data.append([np.array(img_data), my_label(img)])  # Add image data and label to list
    shuffle(data)  # Shuffle the data
    return data
