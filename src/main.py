import numpy as np
import cv2
import imageio
import skimage
from tensorflow.keras.models import load_model
from face_detection import FaceDetector
import os
os.chdir('..')
face_detector = FaceDetector(minsize=100)
crop_ratio = (0.05, 0.05)

class EmotionClassifier:
    def __init__(self, name='mini_xception'):
        self.name = name
        if name == 'simple_cnn':
            self.emotion_classifier = load_model('pretrained_models/fer2013.91-0.68.hdf5', compile=False)
            self.label2index = {'fear': 0, 'happy': 1, 'angry': 2, 'sad': 3, 'surprise': 4, 'disgust': 5, 'neutral': 6}
            self.index2label = {0: 'fear', 1: 'happy', 2: 'angry', 3: 'sad', 4: 'surprise', 5: 'disgust', 6: 'neutral'}
        elif name == 'mini_xception':
            self.emotion_classifier = load_model('pretrained_models/fer2013_mini_XCEPTION.107-0.66.hdf5', compile=False)
            self.label2index = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6}
            self.index2label = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
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

    def classify(self, face):
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray_face = cv2.resize(gray_face, self.input_shape)
        gray_face = self.preprocess_input(gray_face)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_embedding = self.emotion_classifier.predict(gray_face)
        return self.index2label[np.argmax(emotion_embedding[0])]

emotion_classifier = EmotionClassifier()

def process_image(image):
    if image is None :
        return
    image = cv2.GaussianBlur(image,(3, 3),0)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_bbs, _ = face_detector.detect_face(image_rgb)
    for bb in face_bbs:
        bb = [int(x) if x>0 else 0 for x in bb]
        face = image[bb[1]: bb[3], bb[0]: bb[2], :]
        if face is None:
            continue
        label_emotion = emotion_classifier.classify(face)
        cv2.rectangle(image, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0),2)
        front_scale=(bb[3]-bb[0]+bb[2]-bb[1])/130
        thickness =int(front_scale / 4)+1
        if thickness < 1 :
            thickness=1
        if front_scale <1:
            front_scale=1
        cv2.putText(image, label_emotion, (bb[0], bb[3]), cv2.FONT_HERSHEY_PLAIN,front_scale,(255,255,255),thickness,cv2.LINE_AA)
    return image

cap=cv2.VideoCapture(0)
while(cap.isOpened):
    ret, frame = cap.read()
    # flip image
    if frame is None:
        continue
    im =cv2.flip(frame,1)
    im = process_image(im)
    cv2.imshow('frame',im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()