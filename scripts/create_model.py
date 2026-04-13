# ============================================
# СОЗДАНИЕ МОДЕЛИ
# ============================================

import torch
import torch.nn as nn
import torchvision.models as models
import requests
from pathlib import Path

class DishCalorieModel(nn.Module):
    def __init__(self, num_ingredients, hidden_size=1024, dropout=0.3):
        super(DishCalorieModel, self).__init__()
        
        
        # Создаём папку для кэша
        cache_dir = Path.home() / '.cache' / 'torch' / 'hub' / 'checkpoints'
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        model_url = "https://download.pytorch.org/models/resnet50-0676ba61.pth"
        model_path = cache_dir / "resnet50-0676ba61.pth"
        
        # Скачиваем через requests с увеличенным таймаутом
        if not model_path.exists():
            print("Скачивание ResNet50 через requests...")
            response = requests.get(model_url, stream=True, timeout=60)
            response.raise_for_status()
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Скачивание завершено!")
        else:
            print("Модель уже есть в кэше")
        
        # Создаём модель ResNet50
        self.cnn = models.resnet50(weights=None)
        
        # Загружаем веса
        self.cnn.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        # Заменяем fc слой на Identity
        cnn_out_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()
        
        # Размораживаем слои
        for name, param in self.cnn.named_parameters():
            if 'layer4' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # MLP для ингредиентов 
        self.ingredient_fc = nn.Sequential(
            nn.Linear(num_ingredients, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.LayerNorm(512), 
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
        )
        
        # MLP для массы
        self.mass_fc = nn.Sequential(
            nn.Linear(1, 64),
            nn.LayerNorm(64), 
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.LayerNorm(32), 
            nn.ReLU(inplace=True),
        )
        
        # Объединенная сеть 
        self.combined_fc = nn.Sequential(
            nn.Linear(cnn_out_features + 256 + 32, hidden_size),
            nn.LayerNorm(hidden_size), 
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2), 
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 4, 1)
        )
        
    def forward(self, image, ingredients, mass):
        image_features = self.cnn(image)
        ingredient_features = self.ingredient_fc(ingredients)
        mass_features = self.mass_fc(mass)
        combined = torch.cat([image_features, ingredient_features, mass_features], dim=1)
        return self.combined_fc(combined).squeeze()

