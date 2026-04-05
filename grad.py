# grad.py
import cv2
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Load face data and labels
with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Load Haar cascade for face detection
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Threshold for unknown detection
UNKNOWN_THRESHOLD = 10000

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    results = []

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)

        distances, indices = knn.kneighbors(resized_img)
        min_distance = distances[0][0]

        if min_distance < UNKNOWN_THRESHOLD:
            label = knn.predict(resized_img)[0]
        else:
            label = "Unknown Person"

        results.append({
            "label": label,
            "coords": (x, y, w, h)
        })

    return results