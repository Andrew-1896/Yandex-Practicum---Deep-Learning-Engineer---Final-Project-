# ============================================
# КОНФИГ ФАЙЛ
# ============================================

import json
from pathlib import Path
from datetime import datetime


def save_config(config, save_path):
    """
    Сохраняет конфигурацию в JSON файл.
    
    Args:
        config: словарь с конфигурацией
        save_path: путь для сохранения JSON файла
    """
    # Создаём копию конфига, чтобы не изменять оригинал
    config_to_save = config.copy()
    
    # Добавляем timestamp, если его нет
    if 'created_at' not in config_to_save:
        config_to_save['created_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(save_path, 'w') as f:
        json.dump(config_to_save, f, indent=4)
    print(f"✅ Конфигурация сохранена в: {save_path}")


def load_config(config_path):
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


def create_default_config(num_ingredients):
    """
    Создаёт конфигурацию с параметрами по умолчанию.
    
    Args:
        num_ingredients: количество уникальных ингредиентов
        
    Returns:
        dict: конфигурация по умолчанию
    """
    return {
        'model_name': 'DishCalorieModel',
        'batch_size': 64,
        'num_epochs': 60,
        'learning_rate': 5e-4,
        'weight_decay': 5e-5,
        'dropout': 0.2,
        'hidden_size': 1024,
        'num_ingredients': num_ingredients,
        'mixed_precision': True,
        'grad_clip': 1.0
    }


def update_config(config_path, updates):
    """
    Обновляет существующий конфиг новыми параметрами.
    
    Args:
        config_path: путь к JSON файлу
        updates: словарь с обновлениями
        
    Returns:
        dict: обновлённая конфигурация
    """
    config = load_config(config_path)
    config.update(updates)
    save_config(config, config_path)
    return config
