import cv2
import numpy as np
import os
import pickle

# Ensure 'data/' directory exists
if not os.path.exists('data'):
    os.makedirs('data')

video = cv2.VideoCapture(0)    # 0 for webcam
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

face_data = []
i = 0

name = input("Enter your name:")

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        corp_image = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(corp_image, dsize=(50, 50))
        face_data.append(resized_img)  # ✅ Append the face

        # Display count and rectangle
        if len(face_data) <= 100:
            cv2.putText(frame, str(len(face_data)), org=(50, 50),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1,
                        color=(50, 50, 225), thickness=1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 225), 1)

    cv2.imshow("frame", frame)

    k = cv2.waitKey(1)
    if len(face_data) == 100:
        break

video.release()
cv2.destroyAllWindows()

# Save faces in pickle file
face_data = np.array(face_data)
face_data = face_data.reshape(len(face_data), -1)  # ✅ Use actual count

# Save names
if 'names.pkl' not in os.listdir('data/'):
    names = [name] * len(face_data)
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)
    names = names + [name] * len(face_data)
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)  # ✅ Fixed typo here

# Save face data
if 'face_data.pkl' not in os.listdir('data/'):
    with open('data/face_data.pkl', 'wb') as f:
        pickle.dump(face_data, f)
else:
    with open('data/face_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, face_data, axis=0)
    with open('data/face_data.pkl', 'wb') as f:
        pickle.dump(faces, f)
