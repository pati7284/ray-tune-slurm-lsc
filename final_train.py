import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader

def main():
    # 1. Inicjalizacja Weights & Biases
    wandb.init(
        project="pcam-resnet-final", 
        name="resnet-best-params",
        config={
            "learning_rate": 0.000245,
            "batch_size": 16,
            "epochs": 20, # Zwiększona liczba epok
            "architecture": "ResNet18",
            "dataset": "PCAM"
        }
    )
    config = wandb.config

    # 2. Przygotowanie danych (bez ponownego pobierania, korzystamy z dysku SCRATCH)
    data_dir = os.path.join(os.environ.get("SCRATCH", "."), "pcam_data")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.PCAM(root=data_dir, split='train', download=False, transform=transform)
    val_dataset = datasets.PCAM(root=data_dir, split='val', download=False, transform=transform)

    # Ustawiamy num_workers=4, ponieważ SLURM przydzielił nam 4 rdzenie procesora
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    # 3. Inicjalizacja modelu ResNet18
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    print(f"Rozpoczynam finałowy trening na {config.epochs} epok...")

    # 4. Pętla treningowa
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # Zabezpieczenie przed ewentualnym niewłaściwym kształtem etykiet
            labels = labels.squeeze().long() 
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        # Faza walidacji
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                labels = labels.squeeze().long()
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        accuracy = correct / total
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoka [{epoch+1}/{config.epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.4f}")

        # 5. Logowanie W&B - wysyłanie metryk w czasie rzeczywistym
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_accuracy": accuracy
        })

    # 6. Zapisanie ostatecznego modelu na dysk i do chmury
    torch.save(model.state_dict(), "best_resnet_pcam.pth")
    wandb.save("best_resnet_pcam.pth") 
    
    wandb.finish()

if __name__ == "__main__":
    main()
