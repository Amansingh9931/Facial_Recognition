from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime

# try to import Windows TTS; fallback to no-op (ensure function is named 'speak' everywhere)
try:
    from win32com.client import Dispatch
    def speak(text):
        try:
            speaker = Dispatch("SAPI.SpVoice")
            speaker.Speak(text)
        except Exception:
            print("TTS failed:", text)
except Exception:
    def speak(text):
        print("SPEAK:", text)

# determine script base dir so relative data paths work regardless of CWD
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ensure Haar cascade file exists
cascade_file = os.path.join(BASE_DIR, 'data', 'haarcascade_frontalface_default.xml')
if not os.path.exists(cascade_file):
    raise FileNotFoundError(
        f"Haar cascade not found at {cascade_file}. "
        "Run setup_data.py first to download required files."
    )
facedetect = cv2.CascadeClassifier(cascade_file)
if facedetect.empty():
    raise FileNotFoundError(f"Failed to load Haar cascade from {cascade_file}")

# load pickles with clear error messages
names_path = os.path.join(BASE_DIR, 'data', 'names.pkl')
faces_path = os.path.join(BASE_DIR, 'data', 'faces_data.pkl')

if not os.path.exists(names_path) or not os.path.exists(faces_path):
    raise FileNotFoundError("Missing data/*.pkl files. Run add_faces.py to create training data.")

with open(names_path, 'rb') as w:
    LABELS = pickle.load(w)

with open(faces_path, 'rb') as f:
    FACES = pickle.load(f)

# normalize/validate loaded data
if isinstance(FACES, list):
    try:
        FACES = np.vstack([np.asarray(x).reshape(1, -1) for x in FACES])
    except Exception as e:
        raise ValueError("faces_data.pkl content is not in expected format.") from e
else:
    FACES = np.asarray(FACES)

LABELS = np.asarray(LABELS)

if FACES.ndim != 2 or FACES.shape[0] != LABELS.shape[0]:
    raise ValueError("Mismatch between faces and names lengths or unexpected faces shape. "
                     f"faces shape: {FACES.shape}, labels length: {LABELS.shape}")

print('Shape of Faces matrix --> ', FACES.shape)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# load background relative to script and ensure it's large enough for overlay
bg_path = os.path.join(BASE_DIR, "background.png")
imgBackground = cv2.imread(bg_path)
if imgBackground is None:
    # fallback default background: 720x1280 black
    imgBackground = np.zeros((720, 1280, 3), dtype=np.uint8)

# ensure background at least as big as expected overlay (480x640)
min_h, min_w = 480, 640
h_bg, w_bg = imgBackground.shape[:2]
if h_bg < min_h or w_bg < min_w:
    new_h = max(h_bg, min_h)
    new_w = max(w_bg, min_w)
    new_bg = np.zeros((new_h, new_w, 3), dtype=imgBackground.dtype)
    new_bg[0:h_bg, 0:w_bg] = imgBackground
    imgBackground = new_bg

# NEW: initialize camera and attendance directory (use BASE_DIR so paths are consistent)
video = cv2.VideoCapture(0)
if not video.isOpened():
    raise RuntimeError("Cannot open camera. Check camera index (0) and permissions.")

ATT_DIR = os.path.join(BASE_DIR, "Attendance")
os.makedirs(ATT_DIR, exist_ok=True)

COL_NAMES = ['NAME', 'DATE', 'TIME']

# helpers to store latest detection so 'o' key can save attendance even if face not in current frame
last_attendance = None
last_date = None
person_detected = False

while True:
    ret, frame = video.read()
    if not ret or frame is None:
        # camera frame not available, skip this iteration
        continue

    # ensure frame matches expected overlay size (640x480)
    frame_resized = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    person_detected = False
    for (x, y, w, h) in faces:
        # guard crop bounds
        y1, y2 = max(0, y), min(frame_resized.shape[0], y + h)
        x1, x2 = max(0, x), min(frame_resized.shape[1], x + w)
        crop_img = frame_resized[y1:y2, x1:x2, :]
        if crop_img.size == 0:
            continue
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        # ensure dtype matches training
        resized_img = resized_img.astype(FACES.dtype)
        output = knn.predict(resized_img)
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        filename = os.path.join(ATT_DIR, f"Attendance_{date}.csv")
        exist = os.path.isfile(filename)

        # draw boxes and text on the resized frame
        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (50, 50, 255), 2)
        cv2.rectangle(frame_resized, (x1, y1 - 40), (x2, y1), (50, 50, 255), -1)
        cv2.putText(frame_resized, str(output[0]), (x1, y1 - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

        # prepare attendance record for saving on 'o' key (include date)
        last_attendance = [str(output[0]), date, timestamp]
        last_date = date
        person_detected = True
        # break after first detected face to keep behavior similar to original
        break

    # overlay the resized frame on the background at expected position (safe clamped copy)
    frame_h, frame_w = frame_resized.shape[:2]
    y_off, x_off = 162, 55
    h_bg, w_bg = imgBackground.shape[:2]

    # compute overlapping region in background
    y1_bg = max(0, y_off)
    x1_bg = max(0, x_off)
    y2_bg = min(h_bg, y_off + frame_h)
    x2_bg = min(w_bg, x_off + frame_w)

    # corresponding region in frame_resized
    y1_fr = max(0, -y_off)
    x1_fr = max(0, -x_off)
    y2_fr = y1_fr + (y2_bg - y1_bg)
    x2_fr = x1_fr + (x2_bg - x1_bg)

    if (y2_bg > y1_bg) and (x2_bg > x1_bg):
        imgBackground[y1_bg:y2_bg, x1_bg:x2_bg] = frame_resized[y1_fr:y2_fr, x1_fr:x2_fr]
    else:
        # fallback: place resized frame at top-left
        h, w = frame_resized.shape[:2]
        imgBackground[0:h, 0:w] = frame_resized

    cv2.imshow("Frame", imgBackground)
    k = cv2.waitKey(1) & 0xFF

    if k == ord('o'):
        if not person_detected or last_attendance is None or last_date is None:
            speak("No face detected to take attendance.")
            continue

        # use ATT_DIR (created at startup)
        filename = os.path.join(ATT_DIR, f"Attendance_{last_date}.csv")

        speak("Attendance Taken.")
        time.sleep(1)

        # write header if file doesn't exist, otherwise append
        if not os.path.isfile(filename):
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(last_attendance)
        else:
            with open(filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(last_attendance)

    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

# print attendance 
fn = os.path.join(ATT_DIR, f"Attendance_{datetime.now().strftime('%d-%m-%Y')}.csv")
print("File:", fn)
if os.path.exists(fn):
    try:
        with open(fn, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                print(', '.join(row))
    except Exception as e:
        print("Error reading attendance file:", e)
else:
    print("No attendance file found for today.")

