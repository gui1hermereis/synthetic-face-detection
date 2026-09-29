from PIL import Image

from torch.utils.data import Dataset

class DatasetRostos(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        caminho_imagem, rotulo = self.samples[idx]

        with Image.open(caminho_imagem) as img:
            imagem = img.convert("RGB")

        if self.transform:
            imagem = self.transform(imagem)

        return imagem, rotulo