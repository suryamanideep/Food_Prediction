import os
import json
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# Adjust these paths when running if needed
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(ROOT, "data"))
SAVE_DIR = os.environ.get("SAVE_DIR", os.path.join(ROOT, "saved_models"))
os.makedirs(SAVE_DIR, exist_ok=True)




def get_transforms():
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
        [0.229,0.224,0.225])
    ])




def train(epochs=10, batch_size=32, lr=1e-4):
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "test")


    if not os.path.exists(train_dir):
        raise RuntimeError(f"Train directory not found: {train_dir}")


    train_ds = ImageFolder(train_dir, transform=get_transforms())
    val_ds = ImageFolder(val_dir, transform=get_transforms())


    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_b0(pretrained=True)


    # Replace classifier head
    if hasattr(model, "classifier"):
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, len(train_ds.classes))
    else:
        model.classifier = nn.Linear(model.classifier.in_features, len(train_ds.classes))


    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)


    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} train")
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


    # validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                acc = correct / total if total > 0 else 0
                print(f"Validation accuracy: {acc:.4f}")


        # save best
        if acc > best_acc:
            best_acc = acc
            save_path = os.path.join(SAVE_DIR, "food_model_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")


    # save class mapping
    with open(os.path.join(SAVE_DIR, "classes.json"), "w") as f:
        json.dump(train_ds.classes, f)




if __name__ == "__main__":
    train()