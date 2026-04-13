# ============================================
# ФУНКЦИЯ ДЛЯ ВИЗУАЛИЗАЦИИ ПРИМЕРОВ ИЗОБРАЖЕНИЙ
# ============================================

import os
import matplotlib.pyplot as plt
from PIL import Image

def show_sample_images(df, images_dir, num_samples=5):
    sample_df = df.sample(min(num_samples, len(df)))
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    if num_samples == 1:
        axes = [axes]
    
    for idx, (_, row) in enumerate(sample_df.iterrows()):
        img_path = os.path.join(images_dir, row['dish_id'], 'rgb.png')
        if os.path.exists(img_path):
            img = Image.open(img_path)
            axes[idx].imshow(img)
            axes[idx].set_title(f"Калории: {row['total_calories']:.0f}\nМасса: {row['total_mass']}г\nИнгр.: {row['num_ingredients']}", fontsize=10)
        else:
            axes[idx].text(0.5, 0.5, 'Нет изображения', ha='center', va='center')
            axes[idx].set_title(f"ID: {row['dish_id']}")
        axes[idx].axis('off')
    
    plt.suptitle('Примеры изображений блюд', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()