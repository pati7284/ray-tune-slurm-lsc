import torch.nn as nn
from torchvision import models

def get_model():
    """
    Ładuje ResNet18 i modyfikuje ostatnią warstwę pod klasyfikację binarną (PCAM).
    """
    #Pobieramy model 
    model = models.resnet18(weights='DEFAULT')
    
    #Zmieniamy ostatnią warstwę: in_features -> 1 neuron na wyjściu (rak: tak/nie)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    
    return model
