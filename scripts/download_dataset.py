import os
import shutil
import zipfile
import gdown

# ID do arquivo no Google Drive
FILE_ID = "SEU_ID_DO_GOOGLE_DRIVE"

zip_path = "dataset.zip"
dataset_path = "dataset"

# Remove dataset antigo se existir
if os.path.exists(dataset_path):
    print("Removendo dataset antigo...")
    shutil.rmtree(dataset_path)

# Remove zip antigo se existir
if os.path.exists(zip_path):
    print("Removendo zip antigo...")
    os.remove(zip_path)

# Baixa o dataset
print("Baixando dataset...")

url = f"https://drive.google.com/uc?id={FILE_ID}"

gdown.download(
    url,
    zip_path,
    quiet=False
)

# Extrai dataset
print("Extraindo dataset...")

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(".")

# Remove o ZIP
print("Removendo arquivo zip...")

os.remove(zip_path)

# Validação da estrutura
if os.path.exists("dataset/fake") and os.path.exists("dataset/real"):
    print("Dataset pronto!")
    print("Estrutura:")
    print("dataset/")
    print(" ├── fake/")
    print(" └── real/")
else:
    raise Exception("Estrutura do dataset inválida!")