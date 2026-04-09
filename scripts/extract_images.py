import os
import zipfile
import shutil
from PIL import Image

download_dir = "/home/gmreis/synthetic-face-detection/kaggle_data"
final_dir = "/home/gmreis/synthetic-face-detection/dataset"

subsets = {
    "flickrfaceshq-dataset-ffhq": ("real", 1000),
    "person-face-dataset-thispersondoesnotexist": ("fake", 1000)
}

os.makedirs(final_dir, exist_ok=True)

def extract_images(zip_path, extract_to, subset_size=None):
    temp_dir = extract_to + "_tmp"
    os.makedirs(temp_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    files = [f for f in os.listdir(temp_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    files = sorted(files)
    if subset_size:
        files = files[:subset_size]

    os.makedirs(extract_to, exist_ok=True)
    for f in files:
        dest_path = os.path.join(extract_to, f)
        if not os.path.exists(dest_path):  # não sobrescrever
            img = Image.open(os.path.join(temp_dir, f)).convert("RGB")
            img.save(dest_path)

    shutil.rmtree(temp_dir)

# Processar todos os ZIPs
for zip_name, (label, subset_size) in subsets.items():
    zip_file = os.path.join(download_dir, zip_name + ".zip")
    label_dir = os.path.join(final_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    print(f"Extraindo {subset_size} imagens de {zip_name} para {label_dir}...")
    extract_images(zip_file, label_dir, subset_size=subset_size)
    print(f"{label.capitalize()} dataset pronto!\n")