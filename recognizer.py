import argparse
import numpy as np
import cv2
import copy
import os

import model as md
import send

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Arguments
parser = argparse.ArgumentParser(description='Face Emotion Recognition')
parser.add_argument("-m", "--model", action="store", help="path to trained model", default="output/bin/model16-40.weights.h5")
parser.add_argument("-w","--no-window", action="store_true", default=False)
parser.add_argument("-r", "--record", action="store_true", help="Record output video", default=False)
parser.add_argument("-o", "--output", action="store", help="Output filename of video", default="output.mp4")
args = parser.parse_args()

# Model
model = md.getModel()
print(f"Model: {args.model}")
model.load_weights(args.model)

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
# Record
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(args.output, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))
faces = None
gray = None
try:
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
                prediction = model.predict(cropped_img, verbose=0)
                score = 1-prediction[0][0]
                t = f"Anger: {score*10000:.2f}"
                send.send(f"Emotion {score*200:.2f}")
                if score > 0:
                    print(t)
                cv2.putText(frame, t, (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        window = cv2.resize(frame,(width, height), interpolation = cv2.INTER_CUBIC)
        if not args.no_window:
            cv2.imshow('Video', window)
        if args.record:
            out.write(window)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("End")

out.release()
cap.release()
cv2.destroyAllWindows()