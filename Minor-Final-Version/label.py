# label.py

import numpy as np

def my_label(image_name):
    """
    This function assigns labels to images based on the filename.
    It splits the filename to get the name and assigns one-hot encoded labels.
    """
    name = image_name.split('.')[-3]
    if name == "Vaishnavi":
        return np.array([1,0,0])
    elif name == "Khushi":
        return np.array([0,1,0])
    elif name == "Will Smith":
        return np.array([0,0,1])
