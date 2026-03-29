"""
Download dataset from Google Drive to RunPod.

INSTRUCTIONS:
1. Go to your Google Drive folder with the dataset
2. Right-click the folder → Share → Copy link
3. Extract the folder ID from the URL
   URL: https://drive.google.com/drive/folders/XXXXXXXXXXXXX
   Folder ID: XXXXXXXXXXXXX
4. Paste it below
"""

import gdown
import os
import zipfile

# ============================================================
# EDIT THESE VALUES
# ============================================================

# Option A: If your data is in a Google Drive FOLDER
GDRIVE_FOLDER_ID = None

# Option B: If your data is a single ZIP file (N24News dataset)
GDRIVE_FILE_ID = "1OS1fXwZ1Vsj70lEQajccyssxQRYp5X9D"

# ============================================================

DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

if GDRIVE_FILE_ID:
    print("Downloading ZIP file from Google Drive...")
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    output = os.path.join(DATA_DIR, "dataset.zip")
    gdown.download(url, output, quiet=False, fuzzy=True)

    print("Extracting...")
    with zipfile.ZipFile(output, 'r') as z:
        z.extractall(DATA_DIR)
    os.remove(output)
    print(f"Extracted to {DATA_DIR}")

elif GDRIVE_FOLDER_ID and GDRIVE_FOLDER_ID != "YOUR_FOLDER_ID_HERE":
    print("Downloading folder from Google Drive...")
    url = f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}"
    gdown.download_folder(url, output=DATA_DIR, quiet=False)
    print(f"Downloaded to {DATA_DIR}")

else:
    print("ERROR: Please set GDRIVE_FOLDER_ID or GDRIVE_FILE_ID in this script!")
    print("See instructions at the top of the file.")
    exit(1)

# Verify
print("\nFiles in data directory:")
for f in sorted(os.listdir(DATA_DIR)):
    size = os.path.getsize(os.path.join(DATA_DIR, f))
    print(f"  {f} ({size/1024:.0f} KB)")
