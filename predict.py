"""PatchTST 光伏功率预测 - 推理脚本。

提供 predict_future(history_df) 函数:输入 96 行历史数据,返回未来 24 步光伏功率预测(已反标准化)。
模型类定义与 train.py 保持一致。
"""
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# 基础目录(基于本文件位置,避免受运行时 CWD 影响)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
MODEL_PATH = os.path.join(MODEL_DIR, 'patchtst.pth')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'pv_dataset.csv')

# 特征列顺序(与 train.py 一致:y 放第 0 列)
FEATURE_COLS = ['y', 'temperature_2m', 'cloud_cover', 'shortwave_radiation', 'relative_humidity_2m']


# ============ PatchTST 模型(与 train.py 定义一致)============
class PatchTST(nn.Module):
    """自行实现的 PatchTST 模型(channel-independent)。"""

    def __init__(self, n_features=5, seq_len=96, pred_len=24,
                 patch_len=16, stride=8, d_model=64, n_heads=4,
                 num_layers=2, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (seq_len - patch_len) // stride + 1

        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Linear(self.num_patches * d_model, pred_len)
        self.aggregate = nn.Linear(n_features * pred_len, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        batch_size = x.size(0)

        # channel-independent: 每个通道当作独立样本
        x = x.permute(0, 2, 1).contiguous()
        x = x.reshape(batch_size * self.n_features, self.seq_len)

        # Patch 切分
        patches = x.unfold(1, self.patch_len, self.stride)

        # 线性映射 + 位置编码
        embeddings = self.dropout(self.patch_embedding(patches) + self.pos_embedding)

        # Transformer Encoder
        enc_out = self.transformer(embeddings)
        enc_out = enc_out.reshape(batch_size * self.n_features, -1)

        # 每个通道预测 + 汇总
        pred_per_channel = self.head(enc_out)
        pred_per_channel = pred_per_channel.reshape(batch_size, self.n_features, self.pred_len)
        pred_per_channel = pred_per_channel.reshape(batch_size, -1)
        out = self.aggregate(pred_per_channel)
        return out


def predict_future(history_df):
    """输入 96 行历史数据 DataFrame,返回未来 24 步光伏功率预测(已反标准化的真实值)。

    Args:
        history_df: pandas DataFrame,包含 96 行,列同 pv_dataset.csv(含 timestamp 和 5 个特征列)。

    Returns:
        numpy 数组,长度 24,已反标准化的真实功率预测值。
    """
    # 加载 scaler
    with open(SCALER_PATH, 'rb') as f:
        scalers = pickle.load(f)
    feature_scaler = scalers['feature_scaler']
    target_scaler = scalers['target_scaler']

    # 准备输入(列顺序与 train 一致)
    data = history_df[FEATURE_COLS].values.astype(np.float32)
    data_scaled = feature_scaler.transform(data)

    # 加载模型
    model = PatchTST(
        n_features=5, seq_len=96, pred_len=24,
        patch_len=16, stride=8, d_model=64, n_heads=4,
        num_layers=2, dropout=0.1,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    with torch.no_grad():
        x = torch.from_numpy(data_scaled).unsqueeze(0)  # (1, 96, 5)
        pred = model(x).squeeze(0).numpy()  # (24,)

    # 用 target_scaler 反标准化
    pred = target_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    # 物理约束:光伏功率不可能为负
    pred = np.clip(pred, 0, None)
    return pred


if __name__ == '__main__':
    # 命令行演示:从 pv_dataset.csv 读取最后 96 行作为输入,打印预测结果
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    history = df.tail(96).reset_index(drop=True)

    print(f'输入:最后 96 行历史数据(时段 {history["timestamp"].iloc[0]} ~ {history["timestamp"].iloc[-1]})')
    pred = predict_future(history)
    print('未来 24 小时光伏功率预测:')
    for i, v in enumerate(pred):
        print(f'  t+{i + 1:>2}: {v:.4f}')
