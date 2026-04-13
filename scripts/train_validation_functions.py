# ============================================
# ФУНКЦИИ ОБУЧЕНИЯ И ВАЛИДАЦИИ
# ============================================

import numpy as np
import torch
import time
import json
from pathlib import Path
from tqdm import tqdm
from torch.cuda.amp import autocast
from sklearn.metrics import mean_absolute_error, r2_score


# ============================================
# ФУНКЦИЯ ДЛЯ ЗАГРУЗКИ КОНФИГА ИЗ ФАЙЛА
# ============================================

def load_config_from_file(config_path):
    """
    Загружает конфигурацию из JSON файла.
    
    Args:
        config_path: путь к JSON файлу с конфигурацией
        
    Returns:
        dict: загруженная конфигурация
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"✅ Конфигурация загружена из: {config_path}")
    return config


# ============================================
# ОБУЧЕНИЕ ОДНОЙ ЭПОХИ
# ============================================

def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, total_epochs, scaler_target, grad_clip=None):
    """Обучение одной эпохи с прогресс-баром"""
    model.train()
    running_loss = 0.0
    predictions_norm = []
    targets_norm = []
    
    progress_bar = tqdm(
        train_loader, 
        desc=f'Epoch {epoch+1}/{total_epochs} [Train]', 
        unit='batch',
        ncols=100,
        leave=False
    )
    
    for batch in progress_bar:
        images = batch['image'].to(device)
        ingredients = batch['ingredients'].to(device)
        mass = batch['mass'].to(device)  
        calories = batch['calories'].to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images, ingredients, mass)  
            loss = criterion(outputs, calories)
        
        scaler.scale(loss).backward()

        if grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * images.size(0)
        
        predictions_norm.extend(outputs.detach().cpu().flatten().numpy())
        targets_norm.extend(calories.detach().cpu().flatten().numpy())
        
        progress_bar.set_postfix({'loss': f'{loss.item():.2f}'})
    
    epoch_loss = running_loss / len(train_loader.dataset)
    
    predictions = scaler_target.inverse_transform(np.array(predictions_norm).reshape(-1, 1)).flatten()
    targets = scaler_target.inverse_transform(np.array(targets_norm).reshape(-1, 1)).flatten()
    
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    
    return epoch_loss, mae, r2


# ============================================
# ВАЛИДАЦИЯ
# ============================================

def validate(model, val_loader, criterion, device, epoch, total_epochs, scaler_target):
    """Валидация модели с прогресс-баром"""
    model.eval()
    running_loss = 0.0
    predictions_norm = []
    targets_norm = []
    
    progress_bar = tqdm(
        val_loader, 
        desc=f'Epoch {epoch+1}/{total_epochs} [Val]', 
        unit='batch',
        ncols=100,
        leave=False
    )
    
    with torch.no_grad():
        for batch in progress_bar:
            images = batch['image'].to(device)
            ingredients = batch['ingredients'].to(device)
            mass = batch['mass'].to(device)
            calories = batch['calories'].to(device)
            
            with autocast():
                outputs = model(images, ingredients, mass)
                loss = criterion(outputs, calories)
            
            running_loss += loss.item() * images.size(0)
            predictions_norm.extend(outputs.cpu().flatten().numpy())
            targets_norm.extend(calories.cpu().flatten().numpy())
            
            progress_bar.set_postfix({'loss': f'{loss.item():.2f}'})
    
    epoch_loss = running_loss / len(val_loader.dataset)
    
    predictions = scaler_target.inverse_transform(np.array(predictions_norm).reshape(-1, 1)).flatten()
    targets = scaler_target.inverse_transform(np.array(targets_norm).reshape(-1, 1)).flatten()
    
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    
    return epoch_loss, mae, r2, predictions, targets


# ============================================
# ОСНОВНАЯ ФУНКЦИЯ ОБУЧЕНИЯ
# ============================================

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, 
          scaler, device, config, model_save_path, scaler_target, grad_clip=None):
    """
    Основная функция обучения и валидации.
    
    Args:
        config: может быть либо словарём, либо ПУТЁМ К JSON ФАЙЛУ
        model_save_path: путь для сохранения лучшей модели
        scaler_target: StandardScaler для целевой переменной
        grad_clip: градиентное клиппирование (опционально)
        
    Returns:
        dict: история обучения
    """
    
    if isinstance(config, (str, Path)):
        config = load_config_from_file(config)
    
    # Извлекаем параметры из конфига
    num_epochs = config.get('num_epochs', 50)
    batch_size = config.get('batch_size', 64)
    
    history = {
        'train_loss': [], 'train_mae': [], 'train_r2': [],
        'val_loss': [], 'val_mae': [], 'val_r2': []
    }
    
    best_val_mae = float('inf')
    best_epoch = 0
    total_start_time = time.time()
    epoch_times = []
    
    print(f"\n{'='*60}")
    print(f"🚀 НАЧАЛО ОБУЧЕНИЯ")
    print(f"{'='*60}")
    print(f"📋 Параметры обучения из конфига:")
    print(f"   - num_epochs: {num_epochs}")
    print(f"   - batch_size: {batch_size}")
    print(f"   - learning_rate: {config.get('learning_rate', 'N/A')}")
    print(f"   - weight_decay: {config.get('weight_decay', 'N/A')}")
    print(f"   - dropout: {config.get('dropout', 'N/A')}")
    print(f"   - hidden_size: {config.get('hidden_size', 'N/A')}")
    print(f"{'='*60}\n")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Обучение
        train_loss, train_mae, train_r2 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, num_epochs, scaler_target, grad_clip)
        
        # Валидация
        val_loss, val_mae, val_r2, _, _ = validate(
            model, val_loader, criterion, device, epoch, num_epochs, scaler_target
        )
        
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - total_start_time
        epoch_times.append(epoch_time)
        
        # Обновление scheduler
        if scheduler:
            scheduler.step()
        
        # Сохранение истории
        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['train_r2'].append(train_r2)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_r2'].append(val_r2)
        
        # Сохранение лучшей модели
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch + 1
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_mae,
                'val_r2': val_r2,
                'config': config,
                'scaler_target': scaler_target
            }, model_save_path)
        
        # Вывод результатов эпохи
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{num_epochs} | ⏱️  Время эпохи: {epoch_time:.1f}s | Общее время: {total_time/60:.1f}min")
        print(f"{'='*70}")
        print(f"📊 Train - Loss: {train_loss:.4f} | MAE: {train_mae:.2f} | R²: {train_r2:.4f}")
        print(f"🎯 Val   - Loss: {val_loss:.4f} | MAE: {val_mae:.2f} | R²: {val_r2:.4f}")
        
        if val_mae == best_val_mae:
            print(f"🏆 Новая лучшая модель! MAE: {val_mae:.2f}")
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"📈 Learning rate: {current_lr:.2e}")
        
        avg_epoch_time = np.mean(epoch_times[-5:]) if len(epoch_times) >= 5 else np.mean(epoch_times)
        remaining_epochs = num_epochs - (epoch + 1)
        eta = avg_epoch_time * remaining_epochs
        print(f"⏰ Осталось примерно: {eta/60:.1f} минут ({remaining_epochs} эпох)")
        
    print(f"\n{'='*70}")
    print(f"✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print(f"🏆 Лучшая модель: эпоха {best_epoch}, MAE: {best_val_mae:.2f}")
    print(f"💾 Модель сохранена в: {model_save_path}")
    print(f"⏱️  Общее время обучения: {total_time/60:.1f} минут")
    print(f"{'='*70}")
    
    return history