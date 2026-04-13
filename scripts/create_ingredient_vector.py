# ============================================
# ФУНКЦИЯ ДЛЯ СОЗДАНИЯ ВЕКТОРА ИНГРЕДИЕНТОВ
# ============================================

import numpy as np

def get_ingredient_vector(ingredients_list, ingr_to_idx, num_ingredients):
    """
    Создает one-hot вектор ингредиентов на основе названий
    
    Args:
        ingredients_list: список названий ингредиентов (например, ['olive oil', 'salt'])
                         может быть пустым
        ingr_to_idx: словарь для маппинга НАЗВАНИЯ ингредиента -> индекс
                     например, {'olive oil': 0, 'salt': 1, ...}
        num_ingredients: общее количество уникальных ингредиентов
    
    Returns:
        numpy array размером num_ingredients (one-hot вектор)
    """
    vec = np.zeros(num_ingredients, dtype=np.float32)
    
    # Обработка пустого списка
    if not ingredients_list:
        return vec
    
    for ingredient_name in ingredients_list:
        if ingredient_name in ingr_to_idx:
            vec[ingr_to_idx[ingredient_name]] = 1.0
    
    return vec