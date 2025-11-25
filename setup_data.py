import os
import urllib.request
import sys

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
CASCADE_FILE = os.path.join(DATA_DIR, "haarcascade_frontalface_default.xml")

def download_cascade():
    print(f"Downloading Haar cascade to {CASCADE_FILE}...")
    try:
        urllib.request.urlretrieve(CASCADE_URL, CASCADE_FILE)
        print("Download successful!")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False

if __name__ == "__main__":
    if os.path.exists(CASCADE_FILE):
        print(f"Cascade file already exists at {CASCADE_FILE}")
        sys.exit(0)
    
    if download_cascade():
        sys.exit(0)
    sys.exit(1)