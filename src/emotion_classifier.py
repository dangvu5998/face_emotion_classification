from tensorflow.keras.models import load_model
import numpy as np
import cv2

class EmotionClassifier:
    def __init__(self, name='mini_xception', image_color='BGR'):
        self.name = name
        self.image_color = image_color
        if name == 'simple_cnn':
            self.emotion_classifier = load_model('pretrained_models/fer2013.91-0.68.hdf5', compile=False)
            self.label2index = {'fear': 0, 'happy': 1, 'angry': 2, 'sad': 3, 'surprise': 4, 'disgust': 5, 'neutral': 6}
            self.index2label = {0: 'fear', 1: 'happy', 2: 'angry', 3: 'sad', 4: 'surprise', 5: 'disgust', 6: 'neutral'}
        elif name == 'mini_xception':
            self.emotion_classifier = load_model('pretrained_models/fer2013_mini_XCEPTION.107-0.66.hdf5', compile=False)
            self.label2index = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6}
            self.index2label = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
        else:
            raise RuntimeError("Invalid name classifier")
        self.input_shape = self.emotion_classifier.input_shape[1:3]

    def preprocess_input(self, x, v2=True):
        if self.name == 'simple_cnn':
            return x
        x = x.astype('float32')
        x = x / 255.0
        if v2:
            x = x - 0.5
            x = x * 2.0
        return x

    def predict(self, face):
        if self.image_color == 'BGR':
            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face
        gray_face = cv2.resize(gray_face.astype('uint8'), self.input_shape)
        gray_face = self.preprocess_input(gray_face)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_embedding = self.emotion_classifier.predict(gray_face)
        return self.index2label[np.argmax(emotion_embedding[0])]
