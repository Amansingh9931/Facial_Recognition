import cv2
import pickle
import numpy as np
import os
import sys

# Determine folders relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

cascade_path = os.path.join(DATA_DIR, 'haarcascade_frontalface_default.xml')
if not os.path.exists(cascade_path):
    print(f"Missing cascade file: {cascade_path}")
    sys.exit(1)

video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Cannot open camera (cv2.VideoCapture failed).")
    sys.exit(1)

facedetect = cv2.CascadeClassifier(cascade_path)
if facedetect.empty():
    print("Failed to load Haar cascade.")
    sys.exit(1)

faces_data = []
i = 0
name = input("Enter Your Name: ").strip()
if not name:
    print("Empty name entered. Exiting.")
    video.release()
    sys.exit(1)

while True:
    ret, frame = video.read()
    if not ret or frame is None:
        # Skip if frame not read; don't crash
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        # ensure crop indices are within frame bounds
        y1, y2 = max(0, y), min(frame.shape[0], y + h)
        x1, x2 = max(0, x), min(frame.shape[1], x + w)
        crop_img = frame[y1:y2, x1:x2, :]
        if crop_img.size == 0:
            continue
        resized_img = cv2.resize(crop_img, (50, 50))
        # sample every 10 frames until 100 samples
        if len(faces_data) < 100 and i % 10 == 0:
            faces_data.append(resized_img)
        i += 1
        cv2.putText(frame, str(len(faces_data)), (50, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 255), 1)
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q') or len(faces_data) == 100:
        break

video.release()
cv2.destroyAllWindows()

n = len(faces_data)
if n == 0:
    print("No faces captured. Exiting.")
    sys.exit(1)

faces_data = np.asarray(faces_data, dtype=np.uint8)

# If faces_data is 3D (n,50,50,3) reshape to (n, -1); if already 2D, keep as is
if faces_data.ndim >= 3:
    faces_data = faces_data.reshape(n, -1)

names_path = os.path.join(DATA_DIR, 'names.pkl')
faces_path = os.path.join(DATA_DIR, 'faces_data.pkl')

# save names
if not os.path.exists(names_path):
    names = [name] * n
    with open(names_path, 'wb') as f:
        pickle.dump(names, f)
else:
    with open(names_path, 'rb') as f:
        try:
            names = pickle.load(f)
        except Exception:
            names = []
    names = names + [name] * n
    with open(names_path, 'wb') as f:
        pickle.dump(names, f)

# save faces
if not os.path.exists(faces_path):
    with open(faces_path, 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open(faces_path, 'rb') as f:
        try:
            faces = pickle.load(f)
        except Exception:
            faces = np.empty((0, faces_data.shape[1]), dtype=faces_data.dtype)
    faces = np.append(faces, faces_data, axis=0)
    with open(faces_path, 'wb') as f:
        pickle.dump(faces, f)

print(f"Saved {n} face samples for '{name}'.")