import os
import pickle
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
names_path = os.path.join(DATA_DIR, 'names.pkl')
faces_path = os.path.join(DATA_DIR, 'faces_data.pkl')

def load_data():
    """Load and return current data"""
    with open(names_path, 'rb') as f:
        names = pickle.load(f)
    with open(faces_path, 'rb') as f:
        faces = pickle.load(f)
    return names, faces

def save_data(names, faces):
    """Save synchronized data"""
    with open(names_path, 'wb') as f:
        pickle.dump(names, f)
    with open(faces_path, 'wb') as f:
        pickle.dump(faces, f)
    print(f"Saved {len(names)} samples")

def fix_data():
    """Synchronize faces and names data"""
    names, faces = load_data()
    print(f"Before: {len(names)} names, {faces.shape} faces")
    
    # Convert to numpy arrays
    faces = np.asarray(faces)
    names = np.asarray(names)
    
    # Get minimum length
    min_len = min(len(names), faces.shape[0])
    
    # Truncate both to same length
    names = names[:min_len]
    faces = faces[:min_len]
    
    print(f"After: {len(names)} names, {faces.shape} faces")
    
    # Save synchronized data
    save_data(names.tolist(), faces)

if __name__ == "__main__":
    fix_data()