import cv2
import os
import pickle
import numpy as np

# ========= PATH SETUP =========
# Base directory of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directory (for haarcascade and pickle files)
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Haar cascade path (same as in your main file)
cascade_file = os.path.join(DATA_DIR, "haarcascade_frontalface_default.xml")
if not os.path.exists(cascade_file):
    raise FileNotFoundError(
        f"Haar cascade not found at {cascade_file}.\n"
        "Download 'haarcascade_frontalface_default.xml' and place it inside the 'data' folder."
    )

facedetect = cv2.CascadeClassifier(cascade_file)
if facedetect.empty():
    raise FileNotFoundError(f"Failed to load Haar cascade from {cascade_file}")

# Pickle file paths
faces_path = os.path.join(DATA_DIR, "faces_data.pkl")
names_path = os.path.join(DATA_DIR, "names.pkl")

# ========= ASK USER FOR NAME =========
person_name = input("Enter the person's name: ").strip()
if not person_name:
    print("No name entered. Exiting.")
    exit(0)

print(f"Capturing faces for: {person_name}")

# ========= CAMERA SETUP =========
video = cv2.VideoCapture(0)
if not video.isOpened():
    raise RuntimeError("Cannot open camera. Check camera index (0) and permissions.")

# Number of face samples to collect
MAX_SAMPLES = 100  # adjust if you want more/less images

faces_data = []   # will store flattened face images
labels = []       # will store corresponding names (one per image)

count = 0

# ========= CAPTURE LOOP =========
while True:
    ret, frame = video.read()
    if not ret or frame is None:
        print("Failed to capture frame from camera. Skipping...")
        continue

    # Resize frame to match your main script (640x480)
    frame_resized = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Crop face with bounds safety
        y1, y2 = max(0, y), min(frame_resized.shape[0], y + h)
        x1, x2 = max(0, x), min(frame_resized.shape[1], x + w)

        crop_img = frame_resized[y1:y2, x1:x2, :]
        if crop_img.size == 0:
            continue

        # Resize to 50x50 (same as in main script), flatten to 1D
        face_resized = cv2.resize(crop_img, (50, 50))
        face_flatten = face_resized.flatten()

        faces_data.append(face_flatten)
        labels.append(person_name)
        count += 1

        # Draw rectangle and info
        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.rectangle(frame_resized, (x1, y1 - 40), (x2, y1), (50, 50, 255), -1)
        cv2.putText(
            frame_resized,
            f"{person_name} #{count}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_COMPLEX,
            0.8,
            (255, 255, 255),
            1,
        )

        print(f"Captured image {count}/{MAX_SAMPLES} for {person_name}")

        # Stop after collecting enough samples
        if count >= MAX_SAMPLES:
            break

    # Show live video with overlays
    cv2.imshow("Add Faces - Press 'q' to quit", frame_resized)

    # Quit manually if needed
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q") or count >= MAX_SAMPLES:
        break

# ========= CLEANUP CAMERA =========
video.release()
cv2.destroyAllWindows()

if count == 0:
    print("No faces were captured. Exiting without saving.")
    exit(0)

# ========= SAVE / APPEND TO PICKLE FILES =========

# Convert to numpy array
faces_data = np.asarray(faces_data)  # shape: (num_samples, 50*50*3)

# ---- Handle faces_data.pkl ----
if os.path.exists(faces_path):
    with open(faces_path, "rb") as f:
        existing_faces = pickle.load(f)

    if isinstance(existing_faces, list):
        # If it's a list of arrays, extend with new list
        existing_faces.extend(list(faces_data))
        faces_to_save = existing_faces
    else:
        # Assume it's a numpy array
        existing_faces = np.asarray(existing_faces)
        faces_to_save = np.vstack([existing_faces, faces_data])
else:
    # Save as list so your main script can handle both list/array
    faces_to_save = list(faces_data)

with open(faces_path, "wb") as f:
    pickle.dump(faces_to_save, f)

# ---- Handle names.pkl ----
if os.path.exists(names_path):
    with open(names_path, "rb") as f:
        existing_names = pickle.load(f)

    # Ensure list
    if not isinstance(existing_names, list):
        existing_names = list(existing_names)
else:
    existing_names = []

# Extend names list with current person's name repeated for each sample
existing_names.extend(labels)

with open(names_path, "wb") as f:
    pickle.dump(existing_names, f)

print(f"\n✅ Done! Saved {count} face samples for '{person_name}'.")
print(f"faces_data.pkl path: {faces_path}")
print(f"names.pkl path:      {names_path}")
