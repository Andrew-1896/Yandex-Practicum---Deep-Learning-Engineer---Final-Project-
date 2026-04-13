# ============================================
# КЛАСС DishDataset - СОЗДАНИЕ ДАТАСЕТА ДЛЯ БЛЮД С ИЗОБРАЖЕНИЯМИ, ИНГРЕДИЕНТАМИ И МАССОЙ
# ============================================

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class DishDataset(Dataset):
    """Датасет для блюд с изображениями, ингредиентами и массой"""
    
    def __init__(self, df, images_dir, ingredient_vector_col='ingredient_vector_normalized', 
                 target_col='calories_normalized', mass_col='total_mass_normalized', transform=None):
        """
        Args:
            df: DataFrame с данными
            images_dir: путь к папке с изображениями
            ingredient_vector_col: название колонки с вектором ингредиентов
            target_col: название целевой переменной
            mass_col: название колонки с массой
            transform: аугментации для изображений
        """
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.ingredient_vector_col = ingredient_vector_col
        self.target_col = target_col
        self.mass_col = mass_col
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row['dish_id'], 'rgb.png')
        
        if os.path.exists(img_path):
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        
        else:
            # Если изображение отсутствует - выбросить исключение (для отладки)
            raise FileNotFoundError(f"Изображение не найдено: {img_path}")
    
       # Вектор ингредиентов
        ingredient_vector = torch.from_numpy(row[self.ingredient_vector_col]).float()
    
       # Масса
        mass = torch.tensor(row[self.mass_col], dtype=torch.float32).unsqueeze(0)
    
       # Целевая переменная
        calories = torch.tensor(row[self.target_col], dtype=torch.float32)
    
        return {
        'image': image,
        'ingredients': ingredient_vector,
        'mass': mass,
        'calories': calories,
        'dish_id': row['dish_id']
         }



