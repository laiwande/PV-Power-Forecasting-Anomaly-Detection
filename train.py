"""PatchTST 光伏功率预测 - 训练脚本(自行实现 PatchTST,不依赖第三方库)。

含训练曲线可视化与早停机制:训练结束后生成 training_curve.png,
直观展示 train/val loss 走势,便于判断是否过拟合。
"""
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # 无界面后端,适合脚本保存图片
import matplotlib.pyplot as plt

# 中文字体(Windows)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 特征工程模块(扩展特征列与构造函数)
from features import build_features, FEATURE_COLS

# 基础目录(基于本文件位置,避免受运行时 CWD 影响)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'pv_dataset.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
MODEL_PATH = os.path.join(MODEL_DIR, 'patchtst.pth')


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

    # ---- 特征工程:扩展为 16 维特征 ----
    df = build_features(df)
    data = df[FEATURE_COLS].values.astype(np.float32)  # (n_samples, n_features)

    # ---- 保存 16 维特征数据(供分析/可视化使用)----
    FEATURE_PATH = os.path.join(BASE_DIR, 'data', 'pv_dataset_16features.csv')
    df.to_csv(FEATURE_PATH, index=False)
    print(f'16 维特征数据已保存到 {FEATURE_PATH} ({len(df)} 行 × {len(FEATURE_COLS)} 列)')

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

    # ---- 保存 scaler(含特征列清单,供推理时校验)----
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump({
            'feature_scaler': feature_scaler,
            'target_scaler': target_scaler,
            'feature_cols': FEATURE_COLS,
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
        n_features=len(FEATURE_COLS), seq_len=seq_len, pred_len=pred_len,
        patch_len=16, stride=8, d_model=64, n_heads=4,
        num_layers=2, dropout=0.1,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 50          # 最大轮数(有早停兜底,设大点无妨)
    patience = 5         # 早停耐心:验证loss连续 patience 轮不降就停止
    best_val = float('inf')
    best_state = None    # 保存最优模型权重(验证loss最低时)
    patience_counter = 0
    train_history, val_history = [], []  # 记录每轮 loss,用于画曲线

    # ---- 训练循环(含早停)----
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
        train_history.append(last_train)
        val_history.append(last_val)

        # 判断是否为当前最优
        improved = last_val < best_val
        marker = '✓ 最优' if improved else ''
        print(f'Epoch {epoch:02d}/{epochs}  train_loss={last_train:.6f}  val_loss={last_val:.6f}  {marker}')

        if improved:
            best_val = last_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}  # 深拷贝最优权重
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'⏹ 早停触发:验证loss已连续 {patience} 轮未下降,停止训练。')
                break

    # ---- 保存最优模型权重(而非最后一轮,避免过拟合)----
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, MODEL_PATH)
    else:
        torch.save(model.state_dict(), MODEL_PATH)
    print('最优模型权重已保存到', MODEL_PATH)
    print(f'训练完成! 共训练 {len(train_history)} 轮, best_val={best_val:.6f}')

    # ---- 在验证集上计算详细指标并保存 ----
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)  # (n_val_samples, pred_len)
    all_targets = np.concatenate(all_targets, axis=0)

    # 反标准化
    preds_real = target_scaler.inverse_transform(all_preds)
    targets_real = target_scaler.inverse_transform(all_targets)

    # 计算指标
    mae = float(np.mean(np.abs(targets_real - preds_real)))
    rmse = float(np.sqrt(np.mean((targets_real - preds_real) ** 2)))
    # MAPE: 只在真实值 > 0.01 时计算,避免除以接近 0 的值导致爆炸
    mask = targets_real.flatten() > 0.01
    if mask.any():
        mape = float(np.mean(np.abs((targets_real.flatten()[mask] - preds_real.flatten()[mask]) / targets_real.flatten()[mask])) * 100)
    else:
        mape = 0.0
    ss_res = np.sum((targets_real - preds_real) ** 2)
    ss_tot = np.sum((targets_real - np.mean(targets_real)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-8))

    # 保存训练历史 + 指标为 JSON(供前端展示)
    HISTORY_PATH = os.path.join(MODEL_DIR, 'training_history.json')
    history_data = {
        'epochs': list(range(1, len(train_history) + 1)),
        'train_loss': train_history,
        'val_loss': val_history,
        'best_epoch': int(np.argmin(val_history)) + 1,
        'best_val_loss': float(best_val),
        'metrics': {
            'MAE': round(mae, 6),
            'RMSE': round(rmse, 6),
            'MAPE': round(mape, 4),
            'R2': round(r2, 6),
        },
        'config': {
            'seq_len': seq_len,
            'pred_len': pred_len,
            'n_features': len(FEATURE_COLS),
            'batch_size': batch_size,
            'd_model': 64,
            'n_heads': 4,
            'num_layers': 2,
            'dropout': 0.1,
            'patch_len': 16,
            'stride': 8,
            'learning_rate': 1e-3,
            'patience': patience,
            'train_samples': len(train_ds),
            'val_samples': len(val_ds),
        },
        'val_predictions': preds_real.flatten().tolist(),
        'val_targets': targets_real.flatten().tolist(),
    }
    with open(HISTORY_PATH, 'w', encoding='utf-8') as f:
        json.dump(history_data, f, ensure_ascii=False, indent=2)
    print(f'训练历史已保存到 {HISTORY_PATH}')
    print(f'验证集指标: MAE={mae:.6f}  RMSE={rmse:.6f}  MAPE={mape:.2f}%  R²={r2:.6f}')

    # ---- 绘制训练曲线 ----
    CURVE_PATH = os.path.join(BASE_DIR, 'training_curve.png')
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs_x = range(1, len(train_history) + 1)
    ax.plot(epochs_x, train_history, 'o-', color='#4994C4', label='训练损失 train loss', linewidth=2, markersize=5)
    ax.plot(epochs_x, val_history, 's-', color='#065279', label='验证损失 val loss', linewidth=2, markersize=5)
    # 标注最优点
    best_epoch = int(np.argmin(val_history)) + 1
    ax.axvline(x=best_epoch, color='#ef4444', linestyle='--', alpha=0.6, label=f'最优轮次 Epoch {best_epoch}')
    ax.set_xlabel('训练轮次 Epoch', fontsize=13)
    ax.set_ylabel('损失 Loss (MSE)', fontsize=13)
    ax.set_title('PatchTST 训练过程 - 损失曲线', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    # 过拟合区域提示:若验证loss末段上升,加注释
    if len(val_history) > best_epoch:
        ax.annotate('← 验证loss上升,出现过拟合',
                    xy=(best_epoch, val_history[best_epoch - 1]),
                    xytext=(best_epoch + 2, val_history[best_epoch - 1] + 0.02),
                    fontsize=11, color='#ef4444',
                    arrowprops=dict(arrowstyle='->', color='#ef4444'))
    plt.tight_layout()
    plt.savefig(CURVE_PATH, dpi=150)
    print(f'训练曲线已保存到 {CURVE_PATH}')


if __name__ == '__main__':
    main()
