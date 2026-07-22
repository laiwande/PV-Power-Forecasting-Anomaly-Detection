"""光伏功率预测 - 异常检测模块。

基于 Isolation Forest 对"实际功率 vs 预测功率"的残差进行异常检测。

提供 detect_anomalies(actual, predicted) 函数:
    输入等长的实际功率序列与预测功率序列,返回异常标签、异常分数与残差。
    - labels: 1=正常, -1=异常
    - scores: 异常分数(越负越异常)
    - residuals: 残差数组(actual - predicted)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# 复用 predict.py 的路径常量与预测函数
from predict import predict_future, DATA_PATH


def _build_features(residuals, actual):
    """根据残差构造特征矩阵。

    特征包含:
        - 残差绝对值 |residual|
        - 残差百分比 |residual| / (|actual| + 1e-8)(避免除零)
        - 残差本身(带符号,反映高估/低估方向)
        - 滑动窗口残差均值(窗口=3)
        - 滑动窗口残差标准差(窗口=3)
    """
    residuals = np.asarray(residuals, dtype=float)
    actual = np.asarray(actual, dtype=float)
    n = len(residuals)

    # 特征1: 残差绝对值
    abs_res = np.abs(residuals)
    # 特征2: 残差百分比(避免除零)
    pct_res = abs_res / (np.abs(actual) + 1e-8)
    # 特征3: 残差本身(带符号)
    raw_res = residuals

    # 特征4 / 5: 滑动窗口残差均值 / 标准差(窗口=3)
    window = 3
    roll_mean = np.zeros(n)
    roll_std = np.zeros(n)
    for i in range(n):
        start = max(0, i - window + 1)
        seg = residuals[start:i + 1]
        roll_mean[i] = np.mean(seg)
        roll_std[i] = np.std(seg) if len(seg) > 1 else 0.0

    features = np.column_stack([abs_res, pct_res, raw_res, roll_mean, roll_std])
    # 兜底处理可能的 NaN / inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features


def detect_anomalies(actual, predicted, contamination=0.05):
    """对"实际功率 vs 预测功率"做异常检测。

    Args:
        actual: 实际功率序列(numpy 数组或 list)。
        predicted: 预测功率序列(numpy 数组或 list,与 actual 等长)。
        contamination: 异常比例, float 或 'auto'(默认 0.05)。

    Returns:
        dict,包含:
            - labels: 异常标签数组(1=正常, -1=异常),numpy 数组
            - scores: 异常分数数组(越负越异常),numpy 数组
            - residuals: 残差数组(actual - predicted),numpy 数组
            - anomaly_indices: 异常点下标数组(便于上层直接使用)
            - n_anomalies: 异常点数量
            - anomaly_ratio: 异常点占比
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if actual.shape[0] != predicted.shape[0]:
        raise ValueError(
            f'actual 与 predicted 长度不一致: {actual.shape[0]} vs {predicted.shape[0]}'
        )

    n = len(actual)
    # 残差: actual - predicted
    residuals = actual - predicted

    # 样本过少时 IsolationForest 无法稳定拟合,直接判定为全部正常
    if n < 4:
        labels = np.ones(n, dtype=int)
        scores = np.zeros(n, dtype=float)
        anomaly_indices = np.where(labels == -1)[0]
        return {
            'labels': labels,
            'scores': scores,
            'residuals': residuals,
            'anomaly_indices': anomaly_indices,
            'n_anomalies': int(np.sum(labels == -1)),
            'anomaly_ratio': float(np.mean(labels == -1)) if n > 0 else 0.0,
        }

    # 构造残差特征
    features = _build_features(residuals, actual)

    # Isolation Forest 拟合
    iso = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
    )
    labels = iso.fit_predict(features)        # 1=正常, -1=异常
    scores = iso.decision_function(features)  # 越负越异常

    anomaly_indices = np.where(labels == -1)[0]
    return {
        'labels': labels,
        'scores': scores,
        'residuals': residuals,
        'anomaly_indices': anomaly_indices,
        'n_anomalies': int(len(anomaly_indices)),
        'anomaly_ratio': float(len(anomaly_indices) / n) if n > 0 else 0.0,
    }


def _run_sliding_window_demo(df, n_windows=10, start_idx=None):
    """滑动窗口: 多段 96 步历史 -> 预测未来 24 步, 汇总实际/预测功率。

    返回 actual, predicted, timestamps 三个等长 numpy 数组。
    """
    SEQ_LEN = 96
    PRED_LEN = 24
    if start_idx is None:
        # 从数据中段开始取, 避开头部可能的边界
        start_idx = SEQ_LEN

    all_actual = []
    all_predicted = []
    all_ts = []

    for w in range(n_windows):
        hist_start = start_idx + w * PRED_LEN
        if hist_start + SEQ_LEN + PRED_LEN > len(df):
            break
        history = df.iloc[hist_start: hist_start + SEQ_LEN].reset_index(drop=True)
        future = df.iloc[hist_start + SEQ_LEN: hist_start + SEQ_LEN + PRED_LEN]

        actual = future['y'].values.astype(float)
        ts = future['timestamp'].values
        predicted = predict_future(history)

        all_actual.extend(actual)
        all_predicted.extend(predicted)
        all_ts.extend(ts)

    return (
        np.asarray(all_actual, dtype=float),
        np.asarray(all_predicted, dtype=float),
        np.asarray(all_ts),
    )


if __name__ == '__main__':
    # 命令行演示: 滑动窗口方式对多段做预测 + 异常检测
    print('=' * 70)
    print('光伏功率预测 - 异常检测演示(Isolation Forest)')
    print('=' * 70)

    # 1. 读取数据
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f'\n[1] 数据加载完成: 共 {len(df)} 行, 时段 {df["timestamp"].iloc[0]} ~ {df["timestamp"].iloc[-1]}')

    # 2. 滑动窗口收集实际 / 预测功率(10 段, 每段 24 步 = 240 个样本)
    print('\n[2] 滑动窗口预测中(10 段, 每段 96 历史 -> 24 预测)...')
    actual, predicted, ts = _run_sliding_window_demo(df, n_windows=10, start_idx=96)
    print(f'    收集到 {len(actual)} 个样本(实际 vs 预测)')

    # 3. 异常检测
    print('\n[3] 运行 Isolation Forest 异常检测(contamination=0.05)...')
    result = detect_anomalies(actual, predicted, contamination=0.05)
    labels = result['labels']
    scores = result['scores']
    residuals = result['residuals']
    anomaly_idx = result['anomaly_indices']

    # 4. 打印结果
    print('\n[4] 检测结果:')
    print(f'    样本总数      : {len(actual)}')
    print(f'    异常点数量    : {result["n_anomalies"]}')
    print(f'    异常点占比    : {result["anomaly_ratio"] * 100:.2f}%')
    print(f'    残差统计      : mean={np.mean(residuals):.4f}, std={np.std(residuals):.4f}, '
          f'max={np.max(residuals):.4f}, min={np.min(residuals):.4f}')
    print(f'    异常分数统计  : mean={np.mean(scores):.4f}, std={np.std(scores):.4f}, '
          f'max={np.max(scores):.4f}, min={np.min(scores):.4f}')

    if len(anomaly_idx) > 0:
        print('\n    异常点位置(下标 / 时间 / 实际 / 预测 / 残差 / 分数):')
        for i in anomaly_idx:
            ts_str = pd.Timestamp(ts[i]).strftime('%Y-%m-%d %H:%M')
            print(f'      [{i:>3}] {ts_str}  actual={actual[i]:.4f}  '
                  f'pred={predicted[i]:.4f}  resid={residuals[i]:.4f}  score={scores[i]:.4f}')
    else:
        print('\n    未检测到异常点。')

    # 5. 分数样例(展示前 10 个正常点对比)
    print('\n[5] 分数样例(前 10 个样本, 对比正常/异常):')
    for i in range(min(10, len(actual))):
        tag = '异常' if labels[i] == -1 else '正常'
        ts_str = pd.Timestamp(ts[i]).strftime('%Y-%m-%d %H:%M')
        print(f'    [{i:>3}] {ts_str}  {tag}  score={scores[i]:.4f}  resid={residuals[i]:.4f}')

    print('\n' + '=' * 70)
    print('演示完成。')
    print('=' * 70)
