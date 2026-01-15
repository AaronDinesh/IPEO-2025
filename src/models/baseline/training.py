from tqdm import tqdm
import torch
import numpy as np

def train_multi_modal(model, dataloader, val_loader, epochs, optimizer, scheduler, 
        criterion, device, pos_weights: torch.Tensor=None, 
        use_balancing=False, use_bce=True, alpha: float=0.5):
    """
    Train our multi modal fusion baseline model
    """
    
    losses = []
    vlosses = []
    
    for epoch in tqdm(range(epochs), desc='Epoch'):
    #for epoch in range(epochs):
        model.train()
        
        batch_loss = []
        for env_vars, landsat, images, labels in dataloader:
            env_vars.to(device)
            landsat.to(device)
            images.to(device)
            labels.to(device)
            
            output = model(env_vars, landsat, images)
            
            if use_bce:
                loss = criterion(output, labels)   
            
                if use_balancing:
                    cls_loss = combined_losses(output, labels, pos_weight=pos_weights)
                    loss += alpha*cls_loss
            else:
                loss = combined_losses(output, labels, pos_weight=pos_weights)
                
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

                if use_bce:
                    l = criterion(val_predictions, labels)
                    
                    if use_balancing:
                        l += alpha*combined_losses(val_predictions, labels, pos_weight=pos_weights)
                else:
                    l = combined_losses(val_predictions, labels, pos_weight=pos_weights)
                    
                batch_vloss.append(l.item())
            vlosses.append(np.mean(batch_vloss))

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {losses[-1]:.4f} | Val Loss: {vlosses[-1]:.4f}")

    return losses, vlosses


def combined_losses(
    logits, targets,
    pos_weight=None,
    gamma_pos=0.0,
    gamma_neg=4.0,
    clip=0.05,
    reduction="mean",
    eps=1e-8
):

    y = targets.float()
    p = torch.sigmoid(logits)

    # ASL clipping of negative probs
    if clip is not None and clip > 0:
        p_clipped = torch.where(y < 0.5, torch.clamp(p + clip, max=1.0), p)
    else:
        p_clipped = p

    log_p = torch.log(p.clamp(min=eps))
    log_1mp = torch.log((1.0 - p_clipped).clamp(min=eps))

    # focal/asymmetric focusing factors
    w_pos = (1.0 - p).clamp(min=eps).pow(gamma_pos)
    w_neg = p_clipped.clamp(min=eps).pow(gamma_neg)

    # reweight positives by pos_weight
    if pos_weight is not None:
        if pos_weight.ndim == 1:
            pos_w = pos_weight.view(1, -1)
        else:
            pos_w = pos_weight
    else:
        pos_w = 1.0

    # combined loss
    loss_pos = - pos_w * y * w_pos * log_p
    loss_neg = - (1.0 - y) * w_neg * log_1mp
    loss = loss_pos + loss_neg 

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss 