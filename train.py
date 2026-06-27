import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os

def train_epoch(model, dataloader, optimizer, criterion, scaler, scheduler, device, epoch):
    model.train()
    total_loss = 0.0
    
    # Progress bar setup
    is_batch_run = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '') == 'Batch'
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", disable=is_batch_run)
    
    for batch_idx, (input_ids, target_ids) in enumerate(pbar):
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast():
            logits = model(input_ids)
            
            logits = logits.view(-1, logits.size(-1))
            target_ids = target_ids.view(-1)
            
            loss = criterion(logits, target_ids)
            
        scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        scheduler.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item(), 'lr': f"{scheduler.get_last_lr()[0]:.2e}"})
        
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate(model, dataloader, criterion, device, epoch):
    model.eval()
    total_loss = 0.0
    
    is_batch_run = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '') == 'Batch'
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]", disable=is_batch_run)
    
    with torch.no_grad():
        for input_ids, target_ids in pbar:
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            
            with autocast():
                logits = model(input_ids)
                logits = logits.view(-1, logits.size(-1))
                target_ids = target_ids.view(-1)
                
                loss = criterion(logits, target_ids)
                
            total_loss += loss.item()
            pbar.set_postfix({'val_loss': loss.item()})
            
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def train_model(model, train_loader, val_loader, vocab, epochs=10, lr=5e-4, save_dir='/kaggle/working/'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    total_steps = len(train_loader) * epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    
    scaler = GradScaler()
    
    best_val_loss = float('inf')
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Training started on {device}")
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, scheduler, device, epoch)
        val_loss = evaluate(model, val_loader, criterion, device, epoch)
        
        print(f"Epoch {epoch} Summary: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(save_dir, "best_music_transformer.pt")
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved new best model to {save_path} (Val Loss: {best_val_loss:.4f})")
