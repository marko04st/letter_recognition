from matplotlib import pyplot as plt
import pandas as pd
import gzip
import numpy as np
import os
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Input

def load_images(filepath):
    with gzip.open(filepath, 'rb') as f:
        images = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
        images = images.reshape(-1, 28, 28)
        return images

def load_labels(filepath):
    with gzip.open(filepath, 'rb') as f:
        labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    return labels.astype(np.int32)

train_images = load_images('gzip/emnist-letters-train-images-idx3-ubyte.gz')
train_labels = load_labels('gzip/emnist-letters-train-labels-idx1-ubyte.gz')

test_images = load_images('gzip/emnist-letters-test-images-idx3-ubyte.gz')
test_labels = load_labels('gzip/emnist-letters-test-labels-idx1-ubyte.gz')

def transpose_image(img):
    return np.transpose(img)

train_images = np.array([transpose_image(img) for img in train_images])
test_images = np.array([transpose_image(img) for img in test_images])

train_labels -= 1
test_labels -= 1

model_latin = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(26, activation='softmax')
    ])

model_latin.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model_latin.fit(train_images, train_labels, epochs=5)
model_latin.save("model_latin.h5")
test_loss, test_accuracy = model_latin.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_accuracy}')

test_kimages = pd.read_csv('cmnist-master/test.csv')
train_kimages = pd.read_csv('cmnist-master/train.csv')
val_kimages = pd.read_csv('cmnist-master/validation.csv')


KIMAGES_DIR = 'cmnist-master/allbi'

label_column = 'Russian'

train_images_kiril = []
train_labels_kiril= []
for i, row in train_kimages.iterrows():
    img_path = os.path.join(KIMAGES_DIR, row['filename'])
    img = Image.open(img_path).convert('L')
    img = img.resize((28, 28))
    img = np.array(img).astype('float32') / 255.0
    train_images_kiril.append(img)
    train_labels_kiril.append(row[label_column])

train_labels_kiril = np.array(train_labels_kiril).astype(np.int32)
train_images_kiril = np.expand_dims(np.array(train_images_kiril), axis=-1)

test_images_kiril = []
test_labels_kiril= []
for i, row in test_kimages.iterrows():
    img_path = os.path.join(KIMAGES_DIR, row['filename'])
    img = Image.open(img_path).convert('L')
    img = img.resize((28, 28))
    img = np.array(img).astype('float32') / 255.0
    test_images_kiril.append(img)
    test_labels_kiril.append(row[label_column])

test_labels_kiril = np.array(test_labels_kiril).astype(np.int32)
test_images_kiril = np.expand_dims(np.array(test_images_kiril), axis=-1)

mask = train_labels_kiril != -1
train_images_kiril = train_images_kiril[mask]
train_labels_kiril = train_labels_kiril[mask]

mask = test_labels_kiril != -1
test_images_kiril = test_images_kiril[mask]
test_labels_kiril = test_labels_kiril[mask]

unwanted_labels = [6, 28, 30]
print("Unique test labels before filtering:", np.unique(test_labels_kiril))
mask = ~np.isin(train_labels_kiril, unwanted_labels)
train_images_kiril = train_images_kiril[mask]
train_labels_kiril = train_labels_kiril[mask]

mask = ~np.isin(test_labels_kiril, unwanted_labels)
test_images_kiril = test_images_kiril[mask]
test_labels_kiril = test_labels_kiril[mask]

unique_labels = sorted(np.unique(train_labels_kiril))
label_remap = {old: new for new, old in enumerate(unique_labels)}
train_labels_kiril = np.array([label_remap[label] for label in train_labels_kiril])

unique_labels = sorted(np.unique(test_labels_kiril))
label_remap = {old: new for new, old in enumerate(unique_labels)}
test_labels_kiril = np.array([label_remap[label] for label in test_labels_kiril])

print(f"train labels: {len(set(train_labels_kiril))} test labels: {len(set(test_labels_kiril))}")


model_kiril = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(30, activation='softmax')
    ])

model_kiril.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

model_kiril.fit(train_images_kiril, train_labels_kiril, epochs=5)
model_kiril.save("model_kiril.h5")
test_loss,test_accuracy = model_kiril.evaluate(test_images_kiril,test_labels_kiril)
print(f'Test accuracy: {test_accuracy}')
