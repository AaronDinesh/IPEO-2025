from tqdm import tqdm
import torch
import numpy as np

def train_multi_modal(model, dataloader, val_loader, epochs, optimizer, scheduler, criterion, device):
    """
    Train our multi modal fusion baseline model
    """
    
    losses = []
    vlosses = []
    
    for epoch in tqdm(range(epochs), desc='Epoch'):
        model.train()
        
        batch_loss = []
        for env_vars, landsat, images, labels in dataloader:
            env_vars.to(device)
            landsat.to(device)
            images.to(device)
            labels.to(device)
            
            output = model(env_vars, landsat, images)
            
            loss = criterion(output, labels)       
            
            batch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()

        losses.append(np.mean(batch_loss))

        # Validation
        model.eval()
        with torch.no_grad():
            batch_vloss = []
            for env_vars, landsat, images, labels in val_loader:
                env_vars.to(device)
                landsat.to(device)
                images.to(device)
                labels.to(device)

                val_predictions = model(env_vars, landsat, images)

                l = criterion(val_predictions, labels)
                batch_vloss.append(l.item())
            vlosses.append(np.mean(batch_vloss))

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {losses[-1]:.4f} | Val Loss: {vlosses[-1]:.4f}")
        
    return losses, vlosses