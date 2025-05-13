import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from src.data_loader import get_loaders
from src.cnn_model import SimpleCNN

def train_model(epochs=10, lr=1e-3, batch_size=32):
    # Working directory should be project_root
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, train_size, test_size = get_loaders(batch_size)

    model = SimpleCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss, total_correct = 0, 0
        for imgs, labels in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += (outputs.argmax(1) == labels).sum().item()
        train_loss = total_loss / len(train_loader)
        train_acc  = total_correct / train_size * 100

        model.eval()
        val_loss, val_correct = 0, 0
        with torch.inference_mode():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                val_loss += loss_fn(out, labels).item()
                val_correct += (out.argmax(1) == labels).sum().item()
        val_loss /= len(test_loader)
        val_acc   = val_correct / test_size * 100

        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Acc {train_acc:.2f}% | Val Loss {val_loss:.4f}, Acc {val_acc:.2f}%")

    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/model_cnn.pth')


if __name__ == "__main__":
    train_model()