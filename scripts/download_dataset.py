import os
import shutil
import zipfile
import gdown

FILE_ID = "103HKV_9vSH8Ic10cFPgOj1B9y64I3PgH"

# raiz do projeto (um nível acima de scripts)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

zip_path = os.path.join(BASE_DIR, "dataset.zip")
dataset_path = os.path.join(BASE_DIR, "dataset")

# Remove dataset antigo
if os.path.exists(dataset_path):
    print("Removendo dataset antigo...")
    shutil.rmtree(dataset_path)

# Remove zip antigo
if os.path.exists(zip_path):
    print("Removendo zip antigo...")
    os.remove(zip_path)

# Download
print("Baixando dataset...")

url = f"https://drive.google.com/uc?id={FILE_ID}"

gdown.download(
    url,
    zip_path,
    quiet=False
)

# Extração
print("Extraindo dataset...")

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(BASE_DIR)

# Remove ZIP
print("Removendo arquivo zip...")
os.remove(zip_path)

# Validação
fake = os.path.join(dataset_path, "fake")
real = os.path.join(dataset_path, "real")

if os.path.exists(fake) and os.path.exists(real):
    print("\nDataset pronto!")
    print(dataset_path)
    print(" ├── fake/")
    print(" └── real/")
else:
    raise Exception("Estrutura do dataset inválida!")