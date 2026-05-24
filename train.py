import torch
import torch.nn as nn
from ray import tune

#nasze moduły
from dataset import get_dataset
from model import get_model

def train_model(config):
    train_loader, val_loader = get_dataset(config)
    model = get_model()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    #Parametry z configu Optuny
    lr = config.get("lr", 1e-3)
    epochs = config.get("epochs", 5) # Zazwyczaj Ray steruje długością, tu dajemy limit bezpieczeństwa

    criterion = nn.BCEWithLogitsLoss() #funkcja straty dla klasyfikacji binarnej z 1 neuronem na wyjściu
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1) #zmiana całkowitych etykiet bo BCE oczekuje float 
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Zmiana logitów na prawdopodobieństwo (sigmoid) i zaokrąglenie do 0 lub 1
                predicted = torch.round(torch.sigmoid(outputs))
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        avg_val_loss = val_loss / len(val_loader)
        
        #RAPORTOWANIE DO RAYA
        tune.report({"loss": avg_val_loss, "mean_accuracy": accuracy})
