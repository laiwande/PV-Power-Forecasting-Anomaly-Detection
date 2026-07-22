"""光伏功率预测 - 模型评估模块。

对 Persistence(朴素法)、LightGBM、PatchTST 三种模型在验证集上进行滑动窗口评估,
计算 MAE / RMSE / MAPE,打印对比表并生成 eval_results.png 分组柱状图。

评估流程:
1. 加载数据 + build_features 构造 16 维特征
2. 按时间顺序后 20% 作为验证集
3. 在验证集上滑动窗口(步长=24, 限制最多 20 段):96 步历史 -> 24 步预测
4. 三种模型分别预测,收集 actual vs predicted
5. 计算 MAE / RMSE / MAPE(MAPE 只在 actual>0 的样本上计算,分母用 |actual|+1e-8)
6. 打印对比表 + 生成青花瓷配色柱状图 eval_results.png
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互式后端,确保无显示环境下也能保存图片
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor

# 特征工程模块(扩展特征列与构造函数)
from features import build_features, FEATURE_COLS
# PatchTST 推理接口(内部自行处理 scaler 与 build_features)
from predict import predict_future

# 基础目录(基于本文件位置,避免受运行时 CWD 影响)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'pv_dataset.csv')
OUTPUT_FIG = os.path.join(BASE_DIR, 'eval_results.png')

# 评估参数
SEQ_LEN = 96          # 历史窗口长度
PRED_LEN = 24         # 预测长度
STEP = 24             # 滑动步长
MAX_WINDOWS = 20      # 最大评估窗口数(限制 LightGBM 训练 + PatchTST 推理耗时)

# 青花瓷配色(三种模型对应颜色)
COLOR_PERSISTENCE = '#4994C4'   # 孔雀蓝
COLOR_LIGHTGBM    = '#065279'   # 靛蓝
COLOR_PATCHTST    = '#50616D'   # 墨色


# ============ 指标计算 ============
def compute_metrics(actual, predicted):
    """计算 MAE / RMSE / MAPE。

    Args:
        actual: 1D numpy 数组,真实值。
        predicted: 1D numpy 数组,预测值。

    Returns:
        dict: {'mae':..., 'rmse':..., 'mape':...}
        - MAPE 只在 actual>0 的样本上计算,分母用 |actual|+1e-8 避免除零。
        - 若没有 actual>0 的样本,mape 返回 NaN。
    """
    actual = np.asarray(actual, dtype=np.float64).flatten()
    predicted = np.asarray(predicted, dtype=np.float64).flatten()
    # MAE: 平均绝对误差
    mae = float(np.mean(np.abs(predicted - actual)))
    # RMSE: 均方根误差
    rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))
    # MAPE: 平均绝对百分比误差(只在 actual>0 的样本上计算)
    mask = actual > 0
    if mask.sum() > 0:
        mape = float(np.mean(
            np.abs((predicted[mask] - actual[mask]) / (np.abs(actual[mask]) + 1e-8))
        ) * 100.0)
    else:
        mape = float('nan')
    return {'mae': mae, 'rmse': rmse, 'mape': mape}


# ============ Persistence 基线(朴素法)============
def persistence_predict(history_y):
    """朴素法:用历史窗口最后 PRED_LEN 步的 y 值作为预测。

    Args:
        history_y: 1D numpy 数组,长度 >= PRED_LEN,历史 y 值(真实尺度)。

    Returns:
        numpy 数组,长度 PRED_LEN。
    """
    return np.asarray(history_y[-PRED_LEN:], dtype=np.float64).copy()


# ============ LightGBM 基线 ============
def build_lgbm_samples(data, seq_len=SEQ_LEN, pred_len=PRED_LEN, step=STEP):
    """从给定数据(已特征工程)按滑动窗口构造监督学习样本。

    每个样本:
        X = seq_len 步所有特征平铺(seq_len*n_features 维)
        y = 未来 pred_len 步的 y(第 0 列)

    Args:
        data: 2D numpy 数组 (n_rows, n_features),第 0 列为 y。
        seq_len: 历史窗口长度。
        pred_len: 预测长度。
        step: 滑动步长。

    Returns:
        X: (n_samples, seq_len*n_features)
        y: (n_samples, pred_len)
    """
    n_rows, n_features = data.shape
    X, y = [], []
    for start in range(0, n_rows - seq_len - pred_len + 1, step):
        window = data[start:start + seq_len]                              # (seq_len, n_features)
        target = data[start + seq_len:start + seq_len + pred_len, 0]     # (pred_len,)
        X.append(window.flatten())
        y.append(target)
    if not X:
        return np.empty((0, seq_len * n_features)), np.empty((0, pred_len))
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64)


def train_lgbm_model(train_data):
    """训练 LightGBM 多输出回归模型。

    用训练集构造样本,StandardScaler 拟合(在训练集上),MultiOutputRegressor 包装
    LGBMRegressor 预测 24 步。

    Args:
        train_data: 2D numpy 数组 (n_train, n_features),已特征工程,第 0 列为 y。

    Returns:
        dict: {'model':..., 'scaler':...};若 LightGBM 未安装则返回 None。
    """
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        print('[警告] 未安装 lightgbm,跳过 LightGBM 基线。可执行 pip install lightgbm 后重试。')
        return None

    print('[LightGBM] 构造训练样本...')
    X_train, y_train = build_lgbm_samples(train_data)
    print(f'[LightGBM] 训练样本数: {X_train.shape[0]}, 特征维度: {X_train.shape[1]}')

    # StandardScaler 拟合(只用训练集,避免数据泄漏)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print('[LightGBM] 训练 MultiOutputRegressor(LGBMRegressor)...')
    # 基础回归器:轻量级参数,兼顾速度与效果
    base = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    # MultiOutputRegressor 为每个输出步训练一个独立 LightGBM
    model = MultiOutputRegressor(base)
    model.fit(X_train_scaled, y_train)
    print('[LightGBM] 训练完成。')
    return {'model': model, 'scaler': scaler}


def lgbm_predict(lgbm_pack, history_window):
    """用 LightGBM 模型预测未来 PRED_LEN 步。

    Args:
        lgbm_pack: train_lgbm_model 返回的 dict。
        history_window: 2D numpy 数组 (seq_len, n_features),已特征工程,第 0 列为 y。

    Returns:
        numpy 数组,长度 PRED_LEN,真实尺度预测值。
    """
    model = lgbm_pack['model']
    scaler = lgbm_pack['scaler']
    # 96 步历史特征平铺为 1 行向量
    x = history_window.flatten().reshape(1, -1)
    x_scaled = scaler.transform(x)
    pred = model.predict(x_scaled)[0]
    return np.asarray(pred, dtype=np.float64)


# ============ 可视化 ============
def plot_comparison(metrics_dict, save_path=OUTPUT_FIG):
    """绘制三模型 MAE/RMSE/MAPE 分组柱状图(青花瓷配色)。

    Args:
        metrics_dict: {'Persistence': {...}, 'LightGBM': {...}, 'PatchTST': {...}}
        save_path: 图片保存路径。
    """
    # 中文字体(SimHei 优先,回退 Microsoft YaHei)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    models = list(metrics_dict.keys())
    metric_names = ['mae', 'rmse', 'mape']
    metric_labels = ['MAE', 'RMSE', 'MAPE(%)']
    colors = [COLOR_PERSISTENCE, COLOR_LIGHTGBM, COLOR_PATCHTST]

    # 数据矩阵:行=模型,列=指标
    values = np.array(
        [[metrics_dict[m][k] for k in metric_names] for m in models],
        dtype=np.float64,
    )
    # 绘图时将 NaN 置为 0(避免 matplotlib 报错),但在柱子上标注 N/A
    values_plot = np.nan_to_num(values, nan=0.0)

    n_models = len(models)
    x = np.arange(len(metric_names))
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, (model_name, color) in enumerate(zip(models, colors)):
        offset = (i - (n_models - 1) / 2.0) * bar_width
        bars = ax.bar(
            x + offset, values_plot[i], bar_width,
            label=model_name, color=color, edgecolor='white', linewidth=0.8,
        )
        # 在柱子上方标注数值
        for b, v in zip(bars, values[i]):
            if np.isnan(v):
                txt = 'N/A'
            elif v >= 100:
                txt = f'{v:.1f}'
            else:
                txt = f'{v:.3f}'
            ax.text(
                b.get_x() + b.get_width() / 2.0, b.get_height(),
                txt, ha='center', va='bottom', fontsize=8, color='#333333',
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel('指标值', fontsize=11)
    ax.set_title('光伏功率预测模型对比评估', fontsize=14, fontweight='bold', pad=12)
    ax.legend(loc='best', frameon=True, fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    # 顶部留出空间放数值标注
    ymax = values_plot.max() if values_plot.size else 1.0
    ax.set_ylim(0, ymax * 1.18 + 1e-6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'评估图表已保存到: {save_path}')


# ============ 主程序 ============
def main():
    np.random.seed(42)

    print('=' * 60)
    print('光伏功率预测模型对比评估')
    print('=' * 60)

    # ---- 1. 加载数据 + build_features 构造特征 ----
    print('读取数据:', DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    print('构造扩展特征(build_features)...')
    df_feat = build_features(df)
    data_all = df_feat[FEATURE_COLS].values.astype(np.float32)  # (n_total, n_features)

    # ---- 2. 按时间顺序后 20% 作为验证集 ----
    n_total = len(data_all)
    n_train = int(n_total * 0.8)
    train_data = data_all[:n_train]       # 训练集(已特征工程,供 LightGBM 构造样本)
    val_data = data_all[n_train:]         # 验证集(已特征工程,供滑动窗口评估)
    # 原始 df 的验证段(predict_future 内部会调用 build_features,需传入原始 96 行)
    raw_val_df = df.iloc[n_train:].reset_index(drop=True)
    print(f'总样本数: {n_total}, 训练集: {n_train}, 验证集: {n_total - n_train}')

    # ---- 3. 滑动窗口收集评估样本(步长=24, 限制最多 MAX_WINDOWS 段)----
    n_val = len(val_data)
    starts = list(range(0, n_val - SEQ_LEN - PRED_LEN + 1, STEP))
    if len(starts) > MAX_WINDOWS:
        starts = starts[:MAX_WINDOWS]
    print(f'评估窗口数: {len(starts)}(步长 {STEP}, 限制最多 {MAX_WINDOWS})')

    # 收集 actual 和 Persistence 预测(同时保留验证窗口供 LightGBM 使用)
    actuals = []              # list of (PRED_LEN,)
    persistence_preds = []
    val_windows = []          # list of (SEQ_LEN, n_features)
    for start in starts:
        hist = val_data[start:start + SEQ_LEN]                              # (SEQ_LEN, n_features)
        future_y = val_data[start + SEQ_LEN:start + SEQ_LEN + PRED_LEN, 0]  # (PRED_LEN,)
        actuals.append(future_y.astype(np.float64))
        # Persistence: 历史窗口最后 PRED_LEN 步的 y 值
        persistence_preds.append(persistence_predict(hist[:, 0]))
        val_windows.append(hist)
    print('Persistence 预测完成。')

    # ---- 4. LightGBM 训练 + 预测 ----
    lgbm_pack = train_lgbm_model(train_data)
    lgbm_preds = []
    if lgbm_pack is not None:
        for i, hist in enumerate(val_windows):
            pred = lgbm_predict(lgbm_pack, hist.astype(np.float64))
            lgbm_preds.append(pred)
            if (i + 1) % 5 == 0:
                print(f'  LightGBM 预测进度: {i + 1}/{len(val_windows)}')
        print('LightGBM 预测完成。')
    else:
        lgbm_preds = None

    # ---- 5. PatchTST 推理(每段调用一次 predict_future)----
    patchtst_preds = []
    print('PatchTST 推理中(每段调用 predict_future)...')
    for i, start in enumerate(starts):
        # predict_future 内部会 build_features,所以传原始 96 行 DataFrame
        hist_raw = raw_val_df.iloc[start:start + SEQ_LEN].copy()
        try:
            pred = predict_future(hist_raw)
        except Exception as e:
            print(f'[警告] PatchTST 第 {i + 1} 段推理失败: {e},用零向量填充。')
            pred = np.zeros(PRED_LEN, dtype=np.float64)
        patchtst_preds.append(np.asarray(pred, dtype=np.float64))
        if (i + 1) % 5 == 0:
            print(f'  PatchTST 推理进度: {i + 1}/{len(starts)}')
    print('PatchTST 推理完成。')

    # ---- 6. 指标计算(三模型分别计算 MAE/RMSE/MAPE)----
    actual_flat = np.concatenate(actuals)
    persistence_flat = np.concatenate(persistence_preds)

    metrics_dict = {}
    metrics_dict['Persistence'] = compute_metrics(actual_flat, persistence_flat)
    if lgbm_preds is not None:
        lgbm_flat = np.concatenate(lgbm_preds)
        metrics_dict['LightGBM'] = compute_metrics(actual_flat, lgbm_flat)
    else:
        # LightGBM 未安装,占位为 NaN(图表中显示 N/A)
        metrics_dict['LightGBM'] = {'mae': float('nan'), 'rmse': float('nan'), 'mape': float('nan')}
    patchtst_flat = np.concatenate(patchtst_preds)
    metrics_dict['PatchTST'] = compute_metrics(actual_flat, patchtst_flat)

    # ---- 7. 打印对比表 ----
    print()
    print('=' * 60)
    print(f'{"模型":<14}{"MAE":>12}{"RMSE":>12}{"MAPE(%)":>12}')
    print('-' * 60)
    for name, m in metrics_dict.items():
        mae_s = 'N/A' if np.isnan(m['mae']) else f'{m["mae"]:.6f}'
        rmse_s = 'N/A' if np.isnan(m['rmse']) else f'{m["rmse"]:.6f}'
        mape_s = 'N/A' if np.isnan(m['mape']) else f'{m["mape"]:.4f}'
        print(f'{name:<14}{mae_s:>12}{rmse_s:>12}{mape_s:>12}')
    print('=' * 60)

    # ---- 8. 生成图表 ----
    try:
        plot_comparison(metrics_dict, OUTPUT_FIG)
    except Exception as e:
        print(f'[警告] 绘图失败: {e}')

    print('\n评估完成。')


if __name__ == '__main__':
    main()
