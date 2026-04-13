# ============================================
# ФУНКЦИЯ get_data_loaders ДЛЯ ПОЛУЧЕНИЯ DATA LOADER'ОВ
# ============================================

import numpy as np
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from torch.utils.data import DataLoader
from torchvision import transforms
from dish_dataset import DishDataset

def get_data_loaders(dishes_df, images_dir, ingredients_df, batch_size=16, num_workers=4):
    """
    Создает train и val DataLoader'ы
    Предполагает, что векторы ингредиентов УЖЕ созданы в колонке 'ingredient_vector'
    """
    
    # Проверяем, что векторы уже созданы
    if 'ingredient_vector' not in dishes_df.columns:
        raise ValueError("Сначала создайте векторы ингредиентов! Запустите ячейку с one-hot encoding.")
    
    # Разделение по split колонке
    train_df = dishes_df[dishes_df['split'] == 'train'].copy()
    val_df = dishes_df[dishes_df['split'] == 'test'].copy()
    
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    
    # Нормализация one-hot векторов (с помощью MaxAbsScaler)
    ingredient_vectors = np.stack(train_df['ingredient_vector'].values)
    scaler = MaxAbsScaler()
    scaler.fit(ingredient_vectors)
    
    # Применяем нормализацию к данным
    train_df['ingredient_vector_normalized'] = list(scaler.transform(np.stack(train_df['ingredient_vector'].values)))
    val_df['ingredient_vector_normalized'] = list(scaler.transform(np.stack(val_df['ingredient_vector'].values)))
    
    # Нормализация массы
    scaler_mass = StandardScaler()
    train_mass = train_df['total_mass'].values.reshape(-1, 1)
    scaler_mass.fit(train_mass)
    
    train_df['total_mass_normalized'] = scaler_mass.transform(train_mass).flatten()
    val_df['total_mass_normalized'] = scaler_mass.transform(val_df['total_mass'].values.reshape(-1, 1)).flatten()
    
    # Нормализация целевой переменной
    scaler_target = StandardScaler()
    train_targets = train_df['total_calories'].values.reshape(-1, 1)
    scaler_target.fit(train_targets)
    
    train_df['calories_normalized'] = scaler_target.transform(train_targets).flatten()
    val_df['calories_normalized'] = scaler_target.transform(val_df['total_calories'].values.reshape(-1, 1)).flatten()

    IMG_SIZE = 320
    
    # Аугментации для тренировочных изображений
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Трансформации для валидации
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Создание датасетов
    train_dataset = DishDataset(
        train_df, images_dir, 
        ingredient_vector_col='ingredient_vector_normalized',
        target_col='calories_normalized',
        mass_col='total_mass_normalized',
        transform=train_transform
    )
    
    val_dataset = DishDataset(
        val_df, images_dir,
        ingredient_vector_col='ingredient_vector_normalized',
        target_col='calories_normalized',
        mass_col='total_mass_normalized',
        transform=val_transform
    )
    
    # Создание DataLoader'ов
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, scaler, scaler_mass, scaler_target