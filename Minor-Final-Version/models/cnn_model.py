import os
from random import shuffle
from tqdm import tqdm
import cv2
import numpy as np
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore
from sklearn.preprocessing import LabelEncoder

# Specify the full path to the 'data' folder (Update with your actual path)
DATA_PATH = r"C:\Users\rajva\OneDrive\Desktop\Minor-Final-Version\data"
 # Replace with your actual data folder path

def my_label(image_name):
    name = image_name.split('.')[-3] 
    if name == "Vaishnavi":
        return np.array([1,0,0])
    elif name == "Khushi":
        return np.array([0,1,0])
    elif name == "Will Smith":
        return np.array([0,0,1])

def my_data():
    data = []
    # Use the updated DATA_PATH to locate images
    for img in tqdm(os.listdir(DATA_PATH)):
        path = os.path.join(DATA_PATH, img)
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (50, 50))
        data.append([np.array(img_data), my_label(img)])
    shuffle(data)
    return data

data = my_data()
train = data[:360]
test = data[360:]
train = [item for item in train if item[1] is not None]
test = [item for item in test if item[1] is not None]

# Extract features and labels again
X_train = np.array([i[0] for i in train]).reshape(-1, 50, 50, 1)
y_train = np.array([i[1] for i in train])
X_test = np.array([i[0] for i in test]).reshape(-1, 50, 50, 1)
y_test = np.array([i[1] for i in test])

# If the labels are one-hot encoded, convert them to integer labels
y_train = np.argmax(y_train, axis=1)
y_test = np.argmax(y_test, axis=1)

# Define and compile the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # Output layer for 3 classes
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=12, validation_data=(X_test, y_test), verbose=1)

# Display model summary
model.summary()

# Save the model
model.save('model.keras')  # This saves the model to the current working directory
