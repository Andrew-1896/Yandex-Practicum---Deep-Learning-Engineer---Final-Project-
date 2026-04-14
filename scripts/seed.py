# ============================================
# ФУНКЦИЯ ДЛЯ ВОСПРОИЗВОДИМЫХ РЕЗУЛЬТАТОВ
# ============================================

import random
import numpy as np
import torch
import os

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
