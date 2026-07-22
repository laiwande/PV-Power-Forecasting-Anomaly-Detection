"""光伏功率预测 - 特征工程模块。

将原始 5 列特征(timestamp + temperature_2m/cloud_cover/shortwave_radiation/
relative_humidity_2m/y)扩展为 16 维特征,供 PatchTST 模型使用。

新增特征包括:
- solar_elevation: 太阳高度角(纬度 35°N,简化天文公式)
- hour_sin / hour_cos: 小时的时间编码
- month_sin / month_cos: 月份时间编码
- is_daytime: 昼夜标记(短波辐射 > 10 为 1)
- rad_lag1 / rad_lag3 / rad_lag6: 短波辐射滞后特征
- temp_lag3: 温度滞后特征
- rad_ma3: 短波辐射滑动均值
"""
import numpy as np
import pandas as pd

# 纬度常量(35°N)
LATITUDE_DEG = 35.0

# 特征列顺序:y 放第 0 列,方便目标索引;共 16 列
FEATURE_COLS = [
    'y',
    'temperature_2m',
    'cloud_cover',
    'shortwave_radiation',
    'relative_humidity_2m',
    'solar_elevation',
    'hour_sin',
    'hour_cos',
    'month_sin',
    'month_cos',
    'is_daytime',
    'rad_lag1',
    'rad_lag3',
    'rad_lag6',
    'temp_lag3',
    'rad_ma3',
]


def _compute_solar_elevation(timestamps):
    """根据 timestamp 计算太阳高度角(简化天文公式,单位度)。

    Args:
        timestamps: pandas Series,datetime 类型。

    Returns:
        numpy 数组,每个时刻对应的太阳高度角(度)。夜间可能为负值。
    """
    # 纬度转弧度
    lat_rad = np.radians(LATITUDE_DEG)

    # 一年中的第几天(1-365)
    day_of_year = timestamps.dt.dayofyear.values
    # 小时(含小数部分,基于分钟)
    hour = timestamps.dt.hour.values + timestamps.dt.minute.values / 60.0

    # 太阳赤纬 δ(单位:度):简化 Cooper 公式
    # δ = 23.45° * sin(2π * (284 + day_of_year) / 365)
    declination_deg = 23.45 * np.sin(2 * np.pi * (284 + day_of_year) / 365.0)
    declination_rad = np.radians(declination_deg)

    # 时角 H(单位:度):正午 12 点为 0,每小时 15°
    hour_angle_deg = 15.0 * (hour - 12.0)
    hour_angle_rad = np.radians(hour_angle_deg)

    # 太阳高度角 α: sin(α) = sin(φ)sin(δ) + cos(φ)cos(δ)cos(H)
    sin_elevation = (
        np.sin(lat_rad) * np.sin(declination_rad)
        + np.cos(lat_rad) * np.cos(declination_rad) * np.cos(hour_angle_rad)
    )
    # 裁剪到 [-1, 1] 防止浮点误差导致 arcsin 出现 NaN
    sin_elevation = np.clip(sin_elevation, -1.0, 1.0)
    elevation_deg = np.degrees(np.arcsin(sin_elevation))
    return elevation_deg


def build_features(df):
    """对输入 DataFrame 进行特征工程,返回扩展后的 DataFrame。

    输入需包含 timestamp 列和原始 5 列特征:
    y, temperature_2m, cloud_cover, shortwave_radiation, relative_humidity_2m。

    Args:
        df: pandas DataFrame,需包含 timestamp 列(已转为 datetime)和原始 5 列特征。

    Returns:
        新的 DataFrame,包含原始列和新增的 11 个特征列。
    """
    df = df.copy()

    # 确保按时间排序(滞后/滑动特征依赖时间顺序)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

    timestamps = df['timestamp']

    # ---- 太阳高度角 ----
    df['solar_elevation'] = _compute_solar_elevation(timestamps)

    # ---- 时间编码:小时 sin/cos ----
    hour = timestamps.dt.hour.values
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)

    # ---- 时间编码:月份 sin/cos ----
    month = timestamps.dt.month.values
    df['month_sin'] = np.sin(2 * np.pi * month / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * month / 12.0)

    # ---- 昼夜标记:短波辐射 > 10 视为白天 ----
    df['is_daytime'] = (df['shortwave_radiation'] > 10).astype(np.float32)

    # ---- 滞后特征 ----
    # 短波辐射滞后 1/3/6 步
    df['rad_lag1'] = df['shortwave_radiation'].shift(1)
    df['rad_lag3'] = df['shortwave_radiation'].shift(3)
    df['rad_lag6'] = df['shortwave_radiation'].shift(6)
    # 温度滞后 3 步
    df['temp_lag3'] = df['temperature_2m'].shift(3)

    # ---- 滑动均值 ----
    # 短波辐射 3 步滑动均值(min_periods=1 保证开头不产生 NaN)
    df['rad_ma3'] = df['shortwave_radiation'].rolling(window=3, min_periods=1).mean()

    # ---- 后向填充滞后特征开头的 NaN ----
    lag_cols = ['rad_lag1', 'rad_lag3', 'rad_lag6', 'temp_lag3']
    df[lag_cols] = df[lag_cols].bfill()

    return df
