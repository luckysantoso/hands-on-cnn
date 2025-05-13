import os
import zipfile
import requests
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DATA_URL = "https://www.kaggle.com/api/v1/datasets/download/d4rklucif3r/cat-and-dogs"
DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')
EXTRACT_DIR = os.path.join(DATA_ROOT, 'cat-and-dogs')

# Define transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


def download_and_extract():
    os.makedirs(DATA_ROOT, exist_ok=True)
    zip_path = os.path.join(DATA_ROOT, 'cat-and-dogs.zip')
    if not os.path.exists(EXTRACT_DIR):
        print("Downloading dataset...")
        # Use Kaggle API token via environment or config
        r = requests.get(DATA_URL, stream=True)
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(EXTRACT_DIR)
        print("Done.")


def get_loaders(batch_size=32):
    download_and_extract()
    train_dir = os.path.join(EXTRACT_DIR, 'dataset', 'training_set')
    test_dir  = os.path.join(EXTRACT_DIR, 'dataset', 'test_set')
    train_ds = datasets.ImageFolder(root=train_dir, transform=transform)
    test_ds  = datasets.ImageFolder(root=test_dir,  transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, len(train_ds), len(test_ds)

if __name__ == "__main__":
    get_loaders()