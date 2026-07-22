"""光伏功率预测 - LLM 异常诊断模块(mock 占位实现)。

提供 generate_diagnosis(anomaly_info, weather_info) 函数:
    根据异常检测结果与天气信息, 生成中文诊断文本。

当前为模板化 mock 实现, 后续可替换为真实 LLM API 调用(见文件内替换示例)。
"""
import numpy as np
import pandas as pd


def _get_value(obj, keys, default=0.0):
    """从 dict 或对象中按多个候选键名取值(兼容数据集列名与简写)。"""
    # 候选键: 同时支持简写和数据集原始列名
    if isinstance(obj, dict):
        for k in keys:
            if k in obj and obj[k] is not None:
                return float(obj[k])
        return default
    # 对象属性访问: 遍历所有候选键, 命中即返回
    for k in keys:
        if hasattr(obj, k):
            val = getattr(obj, k)
            if val is not None:
                return float(val)
    return default


def _extract_weather(weather_info):
    """从 dict 或 DataFrame 提取天气要素(取均值), 返回标准化的 dict。

    支持的键(兼容简写与数据集列名):
        - temperature: temperature_2m
        - cloud_cover: cloud_cover
        - shortwave_radiation: shortwave_radiation
        - humidity: relative_humidity_2m
    """
    if isinstance(weather_info, pd.DataFrame):
        # DataFrame: 取各列均值
        temp = float(weather_info['temperature_2m'].mean()) if 'temperature_2m' in weather_info else 0.0
        cloud = float(weather_info['cloud_cover'].mean()) if 'cloud_cover' in weather_info else 0.0
        rad = float(weather_info['shortwave_radiation'].mean()) if 'shortwave_radiation' in weather_info else 0.0
        hum = float(weather_info['relative_humidity_2m'].mean()) if 'relative_humidity_2m' in weather_info else 0.0
        return {'temperature': temp, 'cloud_cover': cloud, 'shortwave_radiation': rad, 'humidity': hum}

    # dict / 对象: 兼容多种键名
    temp = _get_value(weather_info, ['temperature', 'temperature_2m', 'temp'])
    cloud = _get_value(weather_info, ['cloud_cover', 'cloud'])
    rad = _get_value(weather_info, ['shortwave_radiation', 'radiation', 'shortwave'])
    hum = _get_value(weather_info, ['humidity', 'relative_humidity_2m', 'rh'])
    return {'temperature': temp, 'cloud_cover': cloud, 'shortwave_radiation': rad, 'humidity': hum}


def _extract_anomaly_info(anomaly_info):
    """从 dict 或对象提取异常统计信息, 返回标准化的 dict。"""
    n_anomalies = _get_value(anomaly_info, ['n_anomalies', 'anomaly_count', 'count'], 0)
    ratio = _get_value(anomaly_info, ['anomaly_ratio', 'ratio'], 0.0)
    # 异常分数统计
    score_min = _get_value(anomaly_info, ['score_min', 'scores_min'], 0.0)
    score_max = _get_value(anomaly_info, ['score_max', 'scores_max'], 0.0)
    score_mean = _get_value(anomaly_info, ['score_mean', 'scores_mean'], 0.0)
    # 残差统计
    resid_mean = _get_value(anomaly_info, ['residual_mean', 'resid_mean', 'residuals_mean'], 0.0)
    resid_std = _get_value(anomaly_info, ['residual_std', 'resid_std', 'residuals_std'], 0.0)
    resid_max = _get_value(anomaly_info, ['residual_max', 'resid_max', 'residuals_max'], 0.0)

    # 如果直接传入了 numpy 数组 / list, 则自行计算统计量
    if isinstance(anomaly_info, dict):
        for arr_key in ('scores', 'residuals'):
            arr = anomaly_info.get(arr_key)
            if isinstance(arr, (np.ndarray, list)) and len(arr) > 0:
                arr = np.asarray(arr, dtype=float)
                if arr_key == 'scores':
                    score_min = float(np.min(arr))
                    score_max = float(np.max(arr))
                    score_mean = float(np.mean(arr))
                else:
                    resid_mean = float(np.mean(arr))
                    resid_std = float(np.std(arr))
                    resid_max = float(np.max(np.abs(arr)))
        # 如果有 labels, 重新计算 n_anomalies / ratio
        labels = anomaly_info.get('labels')
        if isinstance(labels, (np.ndarray, list)) and len(labels) > 0:
            labels_arr = np.asarray(labels)
            n_anomalies = int(np.sum(labels_arr == -1))
            ratio = float(n_anomalies / len(labels_arr))

    return {
        'n_anomalies': int(n_anomalies),
        'anomaly_ratio': float(ratio),
        'score_min': float(score_min),
        'score_max': float(score_max),
        'score_mean': float(score_mean),
        'residual_mean': float(resid_mean),
        'residual_std': float(resid_std),
        'residual_max': float(resid_max),
    }


def _classify_severity(n_anomalies, anomaly_ratio):
    """根据异常点数量与占比划分严重程度。"""
    if n_anomalies == 0:
        return '无异常'
    if anomaly_ratio < 0.1:
        return '轻度'
    if anomaly_ratio < 0.2:
        return '中度'
    return '重度'


def generate_diagnosis(anomaly_info, weather_info):
    """根据异常信息与天气信息生成中文诊断报告(mock 实现)。

    Args:
        anomaly_info: 异常信息, dict 或对象。支持以下键(均可选):
            - n_anomalies / anomaly_count: 异常点数量
            - anomaly_ratio: 异常点占比(0~1)
            - labels: 异常标签数组(1=正常, -1=异常)
            - scores: 异常分数数组
            - residuals: 残差数组
            - score_min / score_max / score_mean: 异常分数统计
            - residual_mean / residual_std / residual_max: 残差统计
        weather_info: 天气信息, dict 或 pandas DataFrame。支持键:
            - temperature / temperature_2m: 温度(°C)
            - cloud_cover: 云量(%)
            - shortwave_radiation: 短波辐射(W/m²)
            - humidity / relative_humidity_2m: 相对湿度(%)

    Returns:
        str: 中文诊断文本。
    """
    info = _extract_anomaly_info(anomaly_info)
    weather = _extract_weather(weather_info)

    n_anomalies = info['n_anomalies']
    ratio = info['anomaly_ratio']
    severity = _classify_severity(n_anomalies, ratio)

    temp = weather['temperature']
    cloud = weather['cloud_cover']
    rad = weather['shortwave_radiation']
    hum = weather['humidity']

    # ===== 以下为 mock 实现, 后续可替换为真实 LLM API 调用 =====

    # 1. 分析天气因素, 收集可能原因
    causes = []
    if rad < 100:
        causes.append(f'短波辐射偏低({rad:.1f} W/m²),太阳能输入不足')
    if cloud > 70:
        causes.append(f'云量较高({cloud:.1f}%),光照被遮挡')
    if hum > 80:
        causes.append(f'湿度偏高({hum:.1f}%),可能为阴雨天气')
    if temp > 35:
        causes.append(f'温度偏高({temp:.1f}°C),组件效率可能下降')
    if temp < -5:
        causes.append(f'温度偏低({temp:.1f}°C),可能影响组件输出')

    weather_abnormal = len(causes) > 0

    # 2. 拼装诊断文本
    lines = []
    lines.append('【诊断报告】')
    lines.append(
        f'检测到 {n_anomalies} 个异常点(占比 {ratio * 100:.2f}%),'
        f'异常严重程度: {severity}。'
    )

    # 异常分数与残差统计
    lines.append(
        f'统计: 异常分数范围 [{info["score_min"]:.4f}, {info["score_max"]:.4f}],'
        f'均值 {info["score_mean"]:.4f}; '
        f'残差均值 {info["residual_mean"]:.4f},标准差 {info["residual_std"]:.4f},'
        f'最大绝对残差 {info["residual_max"]:.4f}。'
    )

    # 天气分析
    lines.append('分析:')
    lines.append(
        f'当前短波辐射为 {rad:.1f} W/m²,云量 {cloud:.1f}%,'
        f'温度 {temp:.1f}°C,相对湿度 {hum:.1f}%。'
    )
    if weather_abnormal:
        lines.append('可能原因:' + ';'.join(causes) + '。')
    else:
        lines.append('当前天气条件整体正常,无明显气象异常。')

    # 3. 结论判断
    if severity == '无异常':
        lines.append('结论:预测功率与实际功率吻合良好,未发现异常波动,系统运行正常。')
    elif weather_abnormal:
        # 有异常且天气不佳 -> 归因于气象
        lines.append(
            '结论:异常主要由气象条件变化引起的光伏输出波动,'
            '属于可解释的正常波动,暂未发现明显设备异常风险。'
        )
    elif severity in ('中度', '重度') and not weather_abnormal:
        # 异常严重但天气正常 -> 提示可能设备异常
        lines.append(
            '结论:在天气条件正常的情况下仍出现较多异常,'
            '可能存在设备异常风险(如组件积灰、遮挡、逆变器故障或传感器异常),'
            '建议进一步排查设备状态与历史告警。'
        )
    else:
        # 轻度异常且天气正常
        lines.append(
            '结论:异常程度较轻且天气正常,'
            '可能与预测模型在个别时段的误差有关,建议持续观察。'
        )

    diagnosis = '\n'.join(lines)

    # ===== mock 实现结束 =====

    # ------------------------------------------------------------------
    # 替换示例(真实 LLM API):
    #   将上面 mock 拼装的 diagnosis 替换为真实大模型生成。
    #   下方 prompt 可作为输入,调用 OpenAI / 智谱 GLM 等接口即可。
    # ------------------------------------------------------------------
    # prompt = (
    #     f"你是光伏运维专家。请根据以下信息生成中文诊断报告:\n"
    #     f"异常信息: 检测到 {n_anomalies} 个异常点(占比 {ratio*100:.2f}%),"
    #     f"严重程度 {severity},异常分数均值 {info['score_mean']:.4f},"
    #     f"残差均值 {info['residual_mean']:.4f},最大残差 {info['residual_max']:.4f}。\n"
    #     f"天气信息: 短波辐射 {rad:.1f} W/m²,云量 {cloud:.1f}%,"
    #     f"温度 {temp:.1f}°C,湿度 {hum:.1f}%。\n"
    #     f"请分析可能原因并给出结论。"
    # )
    #
    # # OpenAI 示例:
    # from openai import OpenAI
    # client = OpenAI(api_key="your-key")
    # response = client.chat.completions.create(
    #     model="gpt-4",
    #     messages=[{"role": "user", "content": prompt}]
    # )
    # return response.choices[0].message.content
    #
    # # 智谱 GLM 示例:
    # from zhipuai import ZhipuAI
    # client = ZhipuAI(api_key="your-key")
    # response = client.chat.completions.create(
    #     model="glm-4",
    #     messages=[{"role": "user", "content": prompt}]
    # )
    # return response.choices[0].message.content
    # ------------------------------------------------------------------

    return diagnosis


def _build_demo_anomaly_info(n_anomalies, ratio):
    """构造演示用 anomaly_info dict(含模拟的 labels / scores / residuals)。"""
    n_total = 24
    labels = np.ones(n_total, dtype=int)
    scores = np.full(n_total, 0.2, dtype=float)
    residuals = np.full(n_total, 0.03, dtype=float)
    # 标记 n_anomalies 个为异常(分数为负, 残差较大)
    n = min(n_anomalies, n_total)
    labels[:n] = -1
    scores[:n] = -0.1
    residuals[:n] = 0.35
    return {
        'n_anomalies': n,
        'anomaly_ratio': ratio,
        'labels': labels,
        'scores': scores,
        'residuals': residuals,
        'score_min': float(np.min(scores)),
        'score_max': float(np.max(scores)),
        'score_mean': float(np.mean(scores)),
        'residual_mean': float(np.mean(residuals)),
        'residual_std': float(np.std(residuals)),
        'residual_max': float(np.max(np.abs(residuals))),
    }


if __name__ == '__main__':
    # 命令行演示: 构造模拟的 anomaly_info 和 weather_info, 生成诊断文本
    print('=' * 70)
    print('光伏功率预测 - LLM 异常诊断演示(mock 实现)')
    print('=' * 70)

    # 场景1: 中度异常 + 天气不佳(短波辐射低 / 云量高 / 湿度高)
    print('\n[场景 1] 中度异常 + 阴雨天气')
    print('-' * 70)
    anomaly_info_1 = _build_demo_anomaly_info(n_anomalies=4, ratio=0.167)
    weather_info_1 = {
        'temperature': 8.5,
        'cloud_cover': 95.0,
        'shortwave_radiation': 35.0,
        'humidity': 90.0,
    }
    diagnosis_1 = generate_diagnosis(anomaly_info_1, weather_info_1)
    print(diagnosis_1)

    # 场景2: 重度异常 + 天气正常(提示可能设备异常)
    print('\n[场景 2] 重度异常 + 天气正常')
    print('-' * 70)
    anomaly_info_2 = _build_demo_anomaly_info(n_anomalies=6, ratio=0.25)
    weather_info_2 = {
        'temperature': 25.0,
        'cloud_cover': 15.0,
        'shortwave_radiation': 650.0,
        'humidity': 45.0,
    }
    diagnosis_2 = generate_diagnosis(anomaly_info_2, weather_info_2)
    print(diagnosis_2)

    # 场景3: 无异常 + 天气正常
    print('\n[场景 3] 无异常 + 天气正常')
    print('-' * 70)
    anomaly_info_3 = _build_demo_anomaly_info(n_anomalies=0, ratio=0.0)
    weather_info_3 = {
        'temperature': 22.0,
        'cloud_cover': 20.0,
        'shortwave_radiation': 520.0,
        'humidity': 50.0,
    }
    diagnosis_3 = generate_diagnosis(anomaly_info_3, weather_info_3)
    print(diagnosis_3)

    # 场景4: 直接传入 numpy 数组(模拟 detect_anomalies 的返回结构子集)
    print('\n[场景 4] 直接传入 labels / scores / residuals 数组')
    print('-' * 70)
    anomaly_info_4 = {
        'labels': np.array([1, 1, -1, -1, 1, 1, 1, 1]),
        'scores': np.array([0.15, 0.12, -0.08, -0.11, 0.18, 0.20, 0.16, 0.14]),
        'residuals': np.array([0.02, -0.01, 0.28, -0.31, 0.03, 0.01, -0.02, 0.04]),
    }
    weather_info_4 = {
        'temperature_2m': 18.0,
        'cloud_cover': 85.0,
        'shortwave_radiation': 80.0,
        'relative_humidity_2m': 85.0,
    }
    diagnosis_4 = generate_diagnosis(anomaly_info_4, weather_info_4)
    print(diagnosis_4)

    print('\n' + '=' * 70)
    print('演示完成。')
    print('=' * 70)
