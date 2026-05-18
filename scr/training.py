import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from preprocessing import AudioPreprocessor
from model import CoughCNN

class CoughDataset(Dataset):
    def __init__(self, data_dir, dataframe):
        self.data_dir = Path(data_dir)
        self.preprocessor = AudioPreprocessor()
        self.samples = []
        for _, row in dataframe.iterrows():
            uuid = row['uuid']
            for ext in ['.wav', '.webm', '.ogg', '.m4a']:
                path = self.data_dir / f"{uuid}{ext}"
                if path.exists():
                    self.samples.append((path, row['label']))
                    break

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        for i in range(len(self.samples)):
            path, label = self.samples[(idx + i) % len(self.samples)]
            tensor = self.preprocessor.process(path)
            if tensor is not None: return tensor, torch.tensor([label], dtype=torch.float32)
        raise RuntimeError("Нет рабочих файлов!")

def prepare_balanced_df(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=['status'])
    df = df[df['status'].isin(['COVID-19', 'healthy']) & (df['cough_detected'] >= 0.8)]
    
    # Балансировка 50/50
    c19 = df[df['status'] == 'COVID-19']
    hlth = df[df['status'] == 'healthy'].sample(n=len(c19), random_state=42)
    df = pd.concat([c19, hlth]).sample(frac=1, random_state=42)
    
    df['label'] = df['status'].map({'COVID-19': 1.0, 'healthy': 0.0})
    return df

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Запуск на: {device}")
    
    base = Path(os.path.abspath(__file__)).parent.parent
    df = prepare_balanced_df(base / "data/metadata_compiled.csv")
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'])
    
    train_loader = DataLoader(CoughDataset(base / "data/raw", train_df), batch_size=16, shuffle=True)
    
    model = CoughCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    crit = nn.BCEWithLogitsLoss()

    for epoch in range(15):
        model.train()
        pbar = tqdm(train_loader, desc=f"Эпоха {epoch+1}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        torch.save(model.state_dict(), base / "models/best_cough_cnn.pth")

if __name__ == "__main__": train()