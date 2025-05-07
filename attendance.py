import cv2
import numpy as np
import os
import csv
import time
import pickle

from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime

# Import from another module (assuming these are defined properly there)
# from main import frame, corp_image, resized_img  # Removed because they're likely unused here

# Start video capture
video = cv2.VideoCapture(0)

# Load face detection model
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load data
with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)

with open('data/face_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Load background image
imgbackground = cv2.imread('bg.jpg')

# Column headers for CSV
COL_NAMES = ['NAME', 'TIME']

# Start main loop
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)

        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        file_path = f'Attendance/Attendance_{date}.csv'
        exist = os.path.isfile(file_path)

        # Drawing on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        attendance = [str(output[0]), str(timestamp)]

    # Combine background and frame
    imgbackground[162:162+480, 55:55+640] = frame

    cv2.imshow("frame", imgbackground)
    k = cv2.waitKey(1)

    if k == ord('o'):
        time.sleep(5)
        with open(file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not exist:
                writer.writerow(COL_NAMES)
            writer.writerow(attendance)

    if k == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()
