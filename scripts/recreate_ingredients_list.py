# ============================================
# ФУНКЦИИ ДЛЯ РАБОТЫ С ID И НАЗВАНИЯМИ ИНГРЕДИЕНТОВ
# ============================================

from typing import List, Dict, Union
import pandas as pd

def extract_numeric_id(ingredient_str: str) -> str:
    """Извлекает числовой ID из строки вида 'ingr_0000000161' или '161'"""
    if ingredient_str.startswith('ingr_'):
        numeric_part = ingredient_str.replace('ingr_', '').lstrip('0')
        if numeric_part == '':
            return '0'
        return numeric_part
    else:
        return str(ingredient_str).lstrip('0')


def id_to_ingredient_name(
    ingredient_id: str, 
    lookup_dict: Dict[Union[int, str], str], 
    default_prefix: str = "unknown"
) -> str:
    """Универсальная функция преобразования ID в название ингредиента
    
    Args:
        ingredient_id: ID ингредиента (например, 'ingr_0000000508', '508', или 508)
        lookup_dict: словарь для преобразования (ключ может быть int или str)
        default_prefix: префикс для неизвестных ингредиентов
    """
    # Извлекаем числовой ID
    if isinstance(ingredient_id, str) and ingredient_id.startswith('ingr_'):
        numeric_part = ingredient_id.replace('ingr_', '').lstrip('0')
        numeric_id = int(numeric_part) if numeric_part else 0
    else:
        # Пробуем преобразовать в int, если это строка с числом
        try:
            numeric_id = int(ingredient_id)
        except (ValueError, TypeError):
            numeric_id = ingredient_id
    
    # Пробуем найти в словаре (сначала как int, потом как str)
    if numeric_id in lookup_dict:
        return lookup_dict[numeric_id]
    elif str(numeric_id) in lookup_dict:
        return lookup_dict[str(numeric_id)]
    
    # Если не нашли, возвращаем fallback значение
    return f"{default_prefix}_{numeric_id}"


def filter_and_convert_ingredients(ingredients_list, id_to_name):
    """
    Фильтрует невалидные ингредиенты и преобразует ID в названия
    
    Args:
        ingredients_list: список ID ингредиентов (например, ['ingr_0000000508', ...])
        id_to_name: словарь для преобразования ID в название
    
    Returns:
        list: список названий ингредиентов
    """
    if not ingredients_list:
        return []
    
    converted = []
    for ingr in ingredients_list:
        # Получаем название ингредиента
        name = id_to_ingredient_name(ingr, id_to_name)
        # Пропускаем deprecated
        if name != 'deprecated':
            converted.append(name)
    
    return converted