import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
from tqdm import tqdm  # <--- ПОЛОСА ПРОГРЕССА

# ПРЯМЫЕ ИМПОРТЫ
from model import CoughCNN
from training import CoughDataset, prepare_dataframe

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CoughCNN().to(device)
    
    base_dir = Path(os.path.abspath(__file__)).parent.parent
    
    model_path = base_dir / "models" / "best_cough_cnn.pth"
    if not model_path.exists():
        print(f"Ошибка: Файл весов {model_path} не найден! Убедитесь, что обучение завершилось.")
        return

    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.eval()

    meta_path = base_dir / "data" / "metadata_compiled.csv"
    data_dir = base_dir / "data" / "raw"

    df = prepare_dataframe(str(meta_path))
    _, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    test_ds = CoughDataset(str(data_dir), test_df)
    
    # УВЕЛИЧИЛИ СКОРОСТЬ В 16 РАЗ (batch_size=16 вместо 1)
    loader = DataLoader(test_ds, batch_size=16, num_workers=0)
    
    y_true, y_pred = [], []
    
    print(f"\nЗапуск тестирования на {len(test_ds)} аудиофайлах...")
    
    with torch.no_grad():
        # Обернули в tqdm для анимации загрузки
        for inputs, labels in tqdm(loader, desc="Оценка", unit="батч"):
            inputs = inputs.to(device)
            out = model(inputs)
            
            # Получаем вероятности и переводим в 0 или 1
            prob = torch.sigmoid(out)
            preds = (prob > 0.5).float()
            
            y_true.extend(labels.cpu().numpy().flatten())
            y_pred.extend(preds.cpu().numpy().flatten())

    print("\n" + "="*50)
    print("ФИНАЛЬНЫЙ ОТЧЕТ НЕЙРОСЕТИ:")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=['Healthy (0)', 'COVID-19 (1)']))

    # Рисуем красивую матрицу ошибок
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Healthy', 'COVID-19'], yticklabels=['Healthy', 'COVID-19'])
    plt.ylabel('Реальный диагноз')
    plt.xlabel('Прогноз нейросети')
    
    docs_dir = base_dir / "docs"
    docs_dir.mkdir(exist_ok=True)
    plt.savefig(str(docs_dir / "confusion_matrix.png"))
    print(f"\nГрафик матрицы ошибок успешно сохранен в папку: {docs_dir.absolute()}")

if __name__ == "__main__":
    evaluate()