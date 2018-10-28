import numpy as np
import cv2
import imageio
import skimage
from face_detection import FaceDetector
import os
from emotion_classifier import EmotionClassifier
os.chdir('..')
face_detector = FaceDetector(minsize=100)
crop_ratio = (0.05, 0.05)

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
        label_emotion = emotion_classifier.predict(face)
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