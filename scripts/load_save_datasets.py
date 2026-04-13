# ============================================
# ФУНКЦИИ ДЛЯ СКАЧИВАНИЯ С ЯНДЕКС.ДИСКА
# ============================================

import os
import zipfile
import requests
from tqdm import tqdm
import shutil

def get_direct_yandex_link(public_url):
    """
    Преобразует публичную ссылку Яндекс.Диска в прямую ссылку на скачивание
    """
    api_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download'
    params = {'public_key': public_url}
    
    print("Получение прямой ссылки для скачивания...")
    response = requests.get(api_url, params=params)
    response.raise_for_status()
    
    direct_link = response.json()['href']
    print("Прямая ссылка получена!")
    return direct_link


def datasets_load_save(yandex_disk_url):
    """
    Скачивает и распаковывает архив с данными с Яндекс.Диска
    Распаковывает так, чтобы папка 'data' оказалась в текущей рабочей папке
    """
    
    archive_path = 'nutrition.zip'  # имя скачиваемого архива
    
    # Проверяем, не распакованы ли уже данные
    if os.path.exists('data') and os.path.exists(os.path.join('data', 'dish.csv')):
        print("✅ Данные уже распакованы в папке 'data'")
        return 'data'
    
    try:
        # Получаем прямую ссылку
        direct_link = get_direct_yandex_link(yandex_disk_url)
        print("Начинается скачивание архива...")
        
        # Скачиваем архив
        response = requests.get(direct_link, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(archive_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Скачивание") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print("✅ Скачивание завершено!")
        
        # Создаём временную папку для распаковки
        temp_dir = '_temp_extract_'
        os.makedirs(temp_dir, exist_ok=True)
        
        # Распаковываем архив во временную папку
        print("Распаковка архива...")
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Находим папку 'data' внутри временной папки
        # (она может быть в корне временной папки или внутри вложенных папок)
        data_source = None
        
        # Ищем папку 'data' рекурсивно
        for root, dirs, files in os.walk(temp_dir):
            if 'data' in dirs:
                data_source = os.path.join(root, 'data')
                break
        
        if data_source and os.path.exists(data_source):
            # Если папка 'data' уже существует в текущей директории, удаляем её
            if os.path.exists('data'):
                shutil.rmtree('data')
            # Перемещаем папку 'data' в текущую директорию
            shutil.move(data_source, 'data')
            print("✅ Папка 'data' перемещена в текущую директорию")
        else:
            raise Exception("Не найдена папка 'data' в архиве")
        
        # Удаляем временную папку и архив
        shutil.rmtree(temp_dir)
        os.remove(archive_path)
        
        print("✅ Распаковка завершена!")
        
        return 'data'
        
    except Exception as e:
        print(f"\n❌ Произошла ошибка: {e}")
        print("\nВозможные решения:")
        print("1. Проверьте интернет-соединение")
        print("2. Скачайте архив вручную по ссылке и распакуйте в текущую папку")
        raise