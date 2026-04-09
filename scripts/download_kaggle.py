import os

os.environ['KAGGLE_CONFIG_DIR'] = "/home/gmreis/synthetic-face-detection/config"
download_dir = "/home/gmreis/synthetic-face-detection/kaggle_data"
os.makedirs(download_dir, exist_ok=True)

datasets = [
    "arnaud58/flickrfaceshq-dataset-ffhq",
    "almightyj/person-face-dataset-thispersondoesnotexist"
]

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
print("Autenticado com sucesso!")

for ds in datasets:
    zip_file = os.path.join(download_dir, ds.split("/")[-1] + ".zip")
    if not os.path.exists(zip_file):
        print(f"Baixando {ds}...")
        api.dataset_download_files(ds, path=download_dir, unzip=False, quiet=False)
    else:
        print(f"{zip_file} já existe, pulando download...")