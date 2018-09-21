import numpy as np
import cv2
import imageio
import skimage
from tensorflow.keras.models import load_model
from face_detection import FaceDetector
import os
os.chdir('..')
emotion_classifier = load_model('pretrained_models/fer2013.91-0.68.hdf5', compile=False)
face_detector = FaceDetector(minsize=100)
label2index = {'fear': 0, 'happy': 1, 'angry': 2, 'sad': 3, 'surprise': 4, 'disgust': 5, 'neutral': 6}
index2label = {0: 'fear', 1: 'happy', 2: 'angry', 3: 'sad', 4: 'surprise', 5: 'disgust', 6: 'neutral'}
crop_ratio = (0.05, 0.05)

def process_image(image):
    if image is None :
        return
    # image = cv2.GaussianBlur(image,(3, 3),0)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_bbs, _ = face_detector.detect_face(image_rgb)
    for bb in face_bbs:
        # face_width = bb[2] - bb[0]
        # face_height = bb[3] - bb[1]
        # offset_width = face_width * crop_ratio[0]
        # offset_height = face_height * crop_ratio[1]
        # bb[0] += offset_width
        # bb[2] -= offset_width
        # bb[1] += offset_height
        # bb[3] -= offset_height
        bb = [int(x) if x>0 else 0 for x in bb]
        face = image[bb[1]: bb[3], bb[0]: bb[2], :]
        if face is None:
            continue
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # face_gray = cv2.bitwise_not(face_gray)
        face_gray = cv2.resize(face_gray, (48, 48))
        face_gray = face_gray.reshape((1, 48, 48, 1))
        emotion_embedding = emotion_classifier.predict(face_gray)
        label_emotion = index2label[np.argmax(emotion_embedding[0])]
        #print(label_emotion)
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
# cap.set(3,3840)#width
# cap.set(4,2160)#height
# cap.set(5,40)#FPS
i=0
while(cap.isOpened):
    i=i+1
    ret, frame = cap.read()
    # flip image
    if frame is None:
        continue
    im =cv2.flip(frame,1)
#     if i%10==0 
    im = process_image(im)
    cv2.imshow('frame',im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()