import numpy as np
import cv2
import copy

import model_cnn
import model_vit
from constants import emotion_dict
from evaluate import evaluate
from train_vit import train_generator

model = model_vit.get_model(train_generator)

model.load_weights('output/vit400.weights.h5')

cascades = [
    'haarcascade_frontalface_default.xml',
    'haarcascade_frontalface_alt.xml',
    'haarcascade_frontalface_alt2.xml',
    'haarcascade_frontalface_alt_tree.xml',
]

width, height = 1280, 720
font_size = 30


# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# start the webcam feed
cap = cv2.VideoCapture(0)
faces = None
gray = None
while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    if not ret:
        break
    index = 0
    nfs = None
    while nfs is None and index < len(cascades):
        classifier = cv2.CascadeClassifier(f'haarcascades/{cascades[index]}')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        nfs = classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        index += 1
    count = sum([1 for _ in nfs])
    if count > 0:
        faces = copy.deepcopy(nfs)
    if faces is not None:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            first = int(np.argmax(prediction))
            second = (np.argsort(prediction)[0][::-1])[1]
            for i in range(len(prediction[0])):
                t = emotion_dict[i] + f": {prediction[0][i]*100:.2f}"
                cv2.putText(frame, t, (0, font_size + i*font_size), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            ev = evaluate(prediction[0])
            t = f"General: {ev*100:.2f}"
            cv2.putText(frame, t, (x-20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            t = f"Cands: {emotion_dict[first]} / {emotion_dict[second]}"
            cv2.putText(frame, t, (x-20, y+h-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Video', cv2.resize(frame,(width, height), interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()