"""PatchTST 光伏功率预测 - 训练脚本(自行实现 PatchTST,不依赖第三方库)"""
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# 基础目录(基于本文件位置,避免受运行时 CWD 影响)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'pv_dataset.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
MODEL_PATH = os.path.join(MODEL_DIR, 'patchtst.pth')

# 特征列顺序:y 放第 0 列,方便目标索引
FEATURE_COLS = ['y', 'temperature_2m', 'cloud_cover', 'shortwave_radiation', 'relative_humidity_2m']


# ============ PatchTST 模型 ============
class PatchTST(nn.Module):
    """自行实现的 PatchTST 模型(channel-independent)。

    流程:输入 (batch, seq_len, n_features)
      -> 按通道独立 reshape 为 (batch*n_features, seq_len)
      -> Patch 切分 -> 线性映射到 embedding
      -> 加入位置编码 -> Transformer Encoder 编码
      -> 展平后线性映射到 pred_len(每个通道各预测一份)
      -> 汇总所有通道 -> 最终 y 预测 (batch, pred_len)
    """

    def __init__(self, n_features=5, seq_len=96, pred_len=24,
                 patch_len=16, stride=8, d_model=64, n_heads=4,
                 num_layers=2, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        # patch 数量:(seq_len - patch_len) // stride + 1
        self.num_patches = (seq_len - patch_len) // stride + 1

        # 线性映射: patch_len -> d_model
        self.patch_embedding = nn.Linear(patch_len, d_model)
        # 可学习的位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 展平后映射到 pred_len(每个通道独立预测)
        self.head = nn.Linear(self.num_patches * d_model, pred_len)
        # 汇总 n_features 个通道的预测 -> 最终 y 预测
        self.aggregate = nn.Linear(n_features * pred_len, pred_len)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        batch_size = x.size(0)

        # channel-independent: 每个通道当作独立样本
        # (batch, seq_len, n_features) -> (batch, n_features, seq_len) -> (batch*n_features, seq_len)
        x = x.permute(0, 2, 1).contiguous()
        x = x.reshape(batch_size * self.n_features, self.seq_len)

        # Patch 切分: (batch*n_features, seq_len) -> (batch*n_features, num_patches, patch_len)
        patches = x.unfold(1, self.patch_len, self.stride)

        # 线性映射到 embedding
        embeddings = self.patch_embedding(patches)  # (B*n_feat, num_patches, d_model)
        # 加入位置编码
        embeddings = self.dropout(embeddings + self.pos_embedding)

        # Transformer Encoder 编码
        enc_out = self.transformer(embeddings)  # (B*n_feat, num_patches, d_model)
        # 展平
        enc_out = enc_out.reshape(batch_size * self.n_features, -1)  # (B*n_feat, num_patches*d_model)
        # 每个通道预测 pred_len
        pred_per_channel = self.head(enc_out)  # (B*n_feat, pred_len)

        # reshape 回 (batch, n_features, pred_len) 后汇总
        pred_per_channel = pred_per_channel.reshape(batch_size, self.n_features, self.pred_len)
        pred_per_channel = pred_per_channel.reshape(batch_size, -1)  # (batch, n_features*pred_len)
        out = self.aggregate(pred_per_channel)  # (batch, pred_len)
        return out


# ============ 数据集 ============
class PVDataset(Dataset):
    """滑窗切片:输入 seq_len 步 -> 目标 pred_len 步(目标只取 y 列,即第 0 列)。"""

    def __init__(self, data, seq_len=96, pred_len=24):
        # data: numpy array (n_samples, 5),已标准化
        self.data = data.astype(np.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]  # (seq_len, 5)
        # 目标只取 y 列(第 0 列),已标准化
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len, 0]  # (pred_len,)
        return torch.from_numpy(x), torch.from_numpy(y)


# ============ 主程序 ============
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ---- 读取数据 ----
    print('读取数据:', DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    data = df[FEATURE_COLS].values.astype(np.float32)  # (n_samples, 5)

    # ---- 划分训练/验证集(按时间顺序,后 20% 作为验证集)----
    n_total = len(data)
    n_train = int(n_total * 0.8)
    train_raw = data[:n_train]
    val_raw = data[n_train:]

    # ---- 标准化(只用训练集拟合)----
    feature_scaler = StandardScaler()
    feature_scaler.fit(train_raw)
    # y 单独 scaler(反标准化时需要)
    target_scaler = StandardScaler()
    target_scaler.fit(train_raw[:, 0:1])

    train_data = feature_scaler.transform(train_raw)
    val_data = feature_scaler.transform(val_raw)

    # ---- 保存 scaler ----
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump({
            'feature_scaler': feature_scaler,
            'target_scaler': target_scaler,
        }, f)
    print('scaler 已保存到', SCALER_PATH)

    # ---- DataLoader ----
    seq_len, pred_len = 96, 24
    batch_size = 64
    train_ds = PVDataset(train_data, seq_len, pred_len)
    val_ds = PVDataset(val_data, seq_len, pred_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    print(f'训练集样本数: {len(train_ds)}, 验证集样本数: {len(val_ds)}')

    # ---- 模型 ----
    device = torch.device('cpu')
    model = PatchTST(
        n_features=5, seq_len=seq_len, pred_len=pred_len,
        patch_len=16, stride=8, d_model=64, n_heads=4,
        num_layers=2, dropout=0.1,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20
    best_val = float('inf')
    last_train, last_val = float('inf'), float('inf')

    # ---- 训练循环 ----
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_losses.append(criterion(pred, y).item())

        last_train = float(np.mean(train_losses))
        last_val = float(np.mean(val_losses))
        print(f'Epoch {epoch:02d}/{epochs}  train_loss={last_train:.6f}  val_loss={last_val:.6f}')

        if last_val < best_val:
            best_val = last_val

    # ---- 保存模型权重(只保存 state_dict)----
    torch.save(model.state_dict(), MODEL_PATH)
    print('模型权重已保存到', MODEL_PATH)
    print(f'训练完成! 最终 train_loss={last_train:.6f}, val_loss={last_val:.6f}, best_val={best_val:.6f}')


if __name__ == '__main__':
    main()
