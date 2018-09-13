import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_fer2013(path):
    '''Load fer2013 emotion classification dataset
    Paramenters:
        path (str): The file location of dataset
    Returns:
        tuple (faces, emotions) includes gray images and list emotions
    '''
    data = pd.read_csv(path)
    pixels = data['pixels'].tolist()
    width = height = 48
    faces = []
    for pixel_sequence in pixels:
        face = np.asarray([int(pixel) for pixel in pixel_sequence.split(' ')]).reshape((height, width))
        faces.append(face)
    emotions = data['emotion']
    index2labels = ['angry', 'disgust', 'fear', 'happy', 'sad',
                'surprise', 'neutral']
    emotion_labels = []
    faces = np.asarray(faces)
    for emotion in emotions:
        emotion_labels.append(index2labels[emotion])
    return faces, emotion_labels

def visualize_data(images, labels, rows=2, cols=5,size_subplot=2):
    num_samples = rows * cols
    plt.figure(figsize=(size_subplot*cols, size_subplot*rows))
    for n, i in enumerate(np.random.randint(len(images), size=rows*cols)):
        plt.subplot(rows, cols, n + 1)
        plt.imshow(images[i], 'gray')
        plt.axis('off')
        plt.title(labels[i])
    plt.show()
