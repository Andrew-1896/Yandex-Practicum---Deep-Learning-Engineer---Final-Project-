# ============================================
# ФУНКЦИЯ ДЛЯ ВОСПРОИЗВОДИМЫХ РЕЗУЛЬТАТОВ
# ============================================

import random
import numpy as np
import torch
import os
from sklearn.preprocessing import StandardScaler

def set_seed(seed=42):
    """Фиксирует seed для воспроизводимости результатов"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Seed установлен на {seed}")

# Нормализует целевую переменную
def normalize_targets(train_df, val_df, target_col='total_calories'):
    """Нормализует целевую переменную"""
    
    scaler_target = StandardScaler()
    train_targets = train_df[target_col].values.reshape(-1, 1)
    scaler_target.fit(train_targets)
    
    train_df['calories_normalized'] = scaler_target.transform(train_targets).flatten()
    val_df['calories_normalized'] = scaler_target.transform(val_df[target_col].values.reshape(-1, 1)).flatten()
    
    return scaler_target