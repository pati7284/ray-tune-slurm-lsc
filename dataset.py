import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_dataset(config):
    #Ścieżka do zapisu danych, dysk SCRATCH
    data_dir = os.path.join(os.environ.get('SCRATCH', './data'), 'pcam_data')
    
    #Transformacje dla zbioru treningowego (z augmentacją)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)), #rozmiar odpowiedni dla ResNet
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Standard ImageNet
    ])

    #Transformacje dla zbioru walidacyjnego (tylko weryfikacja)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.PCAM(root=data_dir, split='train', transform=train_transform, download=True)
    val_dataset = datasets.PCAM(root=data_dir, split='val', transform=val_transform, download=True)

    #batch_size sterowany przez Optunę/Raya
    batch_size = config.get("batch_size", 32)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader
