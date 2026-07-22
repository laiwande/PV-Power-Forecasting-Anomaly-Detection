"""光伏功率预测 - LLM 异常诊断模块(InternLM API 实现)。

提供 generate_diagnosis(anomaly_info, weather_info) 函数:
    根据异常检测结果与天气信息, 调用 InternLM 大模型生成中文诊断报告。

API 配置从 .env 文件读取:
    INTERN_API_KEY: InternLM API Token
    INTERN_API_URL: API 地址(默认 https://chat.intern-ai.org.cn/api/v1/chat/completions)
    INTERN_MODEL:   模型名(默认 intern-latest)

若 API 调用失败(如未配置 Key、网络异常),自动回退到 mock 实现。
"""
import os
import json

import numpy as np
import pandas as pd
import requests

# 尝试加载 .env(使用 python-dotenv;若未安装则回退到 os.environ)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))
except ImportError:
    pass

# InternLM API 配置
API_KEY = os.environ.get('INTERN_API_KEY', '')
API_URL = os.environ.get(
    'INTERN_API_URL',
    'https://chat.intern-ai.org.cn/api/v1/chat/completions',
)
MODEL = os.environ.get('INTERN_MODEL', 'intern-latest')


def _get_value(obj, keys, default=0.0):
    """从 dict 或对象中按多个候选键名取值(兼容数据集列名与简写)。"""
    if isinstance(obj, dict):
        for k in keys:
            if k in obj and obj[k] is not None:
                return float(obj[k])
        return default
    for k in keys:
        if hasattr(obj, k):
            val = getattr(obj, k)
            if val is not None:
                return float(val)
    return default


def _extract_weather(weather_info):
    """从 dict 或 DataFrame 提取天气要素(取均值), 返回标准化的 dict。"""
    if isinstance(weather_info, pd.DataFrame):
        temp = float(weather_info['temperature_2m'].mean()) if 'temperature_2m' in weather_info else 0.0
        cloud = float(weather_info['cloud_cover'].mean()) if 'cloud_cover' in weather_info else 0.0
        rad = float(weather_info['shortwave_radiation'].mean()) if 'shortwave_radiation' in weather_info else 0.0
        hum = float(weather_info['relative_humidity_2m'].mean()) if 'relative_humidity_2m' in weather_info else 0.0
        return {'temperature': temp, 'cloud_cover': cloud, 'shortwave_radiation': rad, 'humidity': hum}

    temp = _get_value(weather_info, ['temperature', 'temperature_2m', 'temp'])
    cloud = _get_value(weather_info, ['cloud_cover', 'cloud'])
    rad = _get_value(weather_info, ['shortwave_radiation', 'radiation', 'shortwave'])
    hum = _get_value(weather_info, ['humidity', 'relative_humidity_2m', 'rh'])
    return {'temperature': temp, 'cloud_cover': cloud, 'shortwave_radiation': rad, 'humidity': hum}


def _extract_anomaly_info(anomaly_info):
    """从 dict 或对象提取异常统计信息, 返回标准化的 dict。"""
    n_anomalies = _get_value(anomaly_info, ['n_anomalies', 'anomaly_count', 'count'], 0)
    ratio = _get_value(anomaly_info, ['anomaly_ratio', 'ratio'], 0.0)
    score_min = _get_value(anomaly_info, ['score_min', 'scores_min'], 0.0)
    score_max = _get_value(anomaly_info, ['score_max', 'scores_max'], 0.0)
    score_mean = _get_value(anomaly_info, ['score_mean', 'scores_mean'], 0.0)
    resid_mean = _get_value(anomaly_info, ['residual_mean', 'resid_mean', 'residuals_mean'], 0.0)
    resid_std = _get_value(anomaly_info, ['residual_std', 'resid_std', 'residuals_std'], 0.0)
    resid_max = _get_value(anomaly_info, ['residual_max', 'resid_max', 'residuals_max'], 0.0)

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


def _build_prompt(info, weather, severity):
    """构造发送给 InternLM 的诊断 prompt。"""
    n_anomalies = info['n_anomalies']
    ratio = info['anomaly_ratio']
    temp = weather['temperature']
    cloud = weather['cloud_cover']
    rad = weather['shortwave_radiation']
    hum = weather['humidity']

    prompt = (
        f"你是光伏电站运维专家。请根据以下光伏电站运行数据,生成一份中文异常诊断报告。\n\n"
        f"【异常检测结果】\n"
        f"- 异常点数量: {n_anomalies} 个\n"
        f"- 异常点占比: {ratio * 100:.2f}%\n"
        f"- 异常严重程度: {severity}\n"
        f"- 异常分数范围: [{info['score_min']:.4f}, {info['score_max']:.4f}],均值 {info['score_mean']:.4f}\n"
        f"- 残差(实际功率 - 预测功率)统计: 均值 {info['residual_mean']:.4f},"
        f"标准差 {info['residual_std']:.4f},最大绝对残差 {info['residual_max']:.4f}\n\n"
        f"【天气环境数据(异常时段均值)】\n"
        f"- 2米高度气温: {temp:.1f}°C\n"
        f"- 云量: {cloud:.1f}%\n"
        f"- 短波太阳辐射: {rad:.1f} W/m²\n"
        f"- 相对湿度: {hum:.1f}%\n\n"
        f"【要求】\n"
        f"1. 分析异常可能的原因(结合天气因素:短波辐射、云量、温度、湿度)\n"
        f"2. 判断异常属于气象条件引起的正常波动,还是可能的设备异常风险\n"
        f"3. 给出明确的结论和建议\n"
        f"4. 报告格式:用【诊断报告】开头,分'异常概述'、'原因分析'、'结论与建议'三部分\n"
        f"5. 语言简洁专业,总字数 200-400 字"
    )
    return prompt


def _call_intern_api(prompt):
    """调用 InternLM API 生成诊断文本。

    Returns:
        str: 模型生成的诊断文本;若调用失败返回 None。
    """
    if not API_KEY:
        print('[LLM 诊断] 未配置 INTERN_API_KEY,将使用 mock 实现。')
        return None

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}',
    }
    payload = {
        'model': MODEL,
        'messages': [{'role': 'user', 'content': prompt}],
        'n': 1,
        'temperature': 0.7,
        'top_p': 0.9,
    }

    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=120)
        if response.status_code != 200:
            print(f'[LLM 诊断] API 返回状态码 {response.status_code}: {response.text[:200]}')
            return None
        result = response.json()
        content = result['choices'][0]['message']['content']
        return content.strip()
    except Exception as e:
        print(f'[LLM 诊断] API 调用异常: {e}')
        return None


def _mock_diagnosis(info, weather, severity):
    """mock 兜底实现: API 不可用时生成模板化诊断文本。"""
    n_anomalies = info['n_anomalies']
    ratio = info['anomaly_ratio']
    temp = weather['temperature']
    cloud = weather['cloud_cover']
    rad = weather['shortwave_radiation']
    hum = weather['humidity']

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
    lines = []
    lines.append('【诊断报告】(mock 兜底)')
    lines.append(f'检测到 {n_anomalies} 个异常点(占比 {ratio * 100:.2f}%),异常严重程度: {severity}。')
    lines.append(
        f'统计: 异常分数范围 [{info["score_min"]:.4f}, {info["score_max"]:.4f}],'
        f'均值 {info["score_mean"]:.4f}; '
        f'残差均值 {info["residual_mean"]:.4f},标准差 {info["residual_std"]:.4f},'
        f'最大绝对残差 {info["residual_max"]:.4f}。'
    )
    lines.append('分析:')
    lines.append(f'当前短波辐射为 {rad:.1f} W/m²,云量 {cloud:.1f}%,温度 {temp:.1f}°C,相对湿度 {hum:.1f}%。')
    if weather_abnormal:
        lines.append('可能原因:' + ';'.join(causes) + '。')
    else:
        lines.append('当前天气条件整体正常,无明显气象异常。')

    if severity == '无异常':
        lines.append('结论:预测功率与实际功率吻合良好,未发现异常波动,系统运行正常。')
    elif weather_abnormal:
        lines.append('结论:异常主要由气象条件变化引起的光伏输出波动,属于可解释的正常波动,暂未发现明显设备异常风险。')
    elif severity in ('中度', '重度') and not weather_abnormal:
        lines.append('结论:在天气条件正常的情况下仍出现较多异常,可能存在设备异常风险(如组件积灰、遮挡、逆变器故障或传感器异常),建议进一步排查设备状态与历史告警。')
    else:
        lines.append('结论:异常程度较轻且天气正常,可能与预测模型在个别时段的误差有关,建议持续观察。')

    return '\n'.join(lines)


def generate_diagnosis(anomaly_info, weather_info):
    """根据异常信息与天气信息生成中文诊断报告。

    优先调用 InternLM API;若 API 不可用则回退到 mock 实现。

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
    severity = _classify_severity(info['n_anomalies'], info['anomaly_ratio'])

    # 构造 prompt 并调用 InternLM API
    prompt = _build_prompt(info, weather, severity)
    diagnosis = _call_intern_api(prompt)

    # API 调用失败则回退到 mock
    if diagnosis is None:
        diagnosis = _mock_diagnosis(info, weather, severity)

    return diagnosis


def _build_demo_anomaly_info(n_anomalies, ratio):
    """构造演示用 anomaly_info dict(含模拟的 labels / scores / residuals)。"""
    n_total = 24
    labels = np.ones(n_total, dtype=int)
    scores = np.full(n_total, 0.2, dtype=float)
    residuals = np.full(n_total, 0.03, dtype=float)
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
    # 命令行演示: 调用 InternLM API 生成诊断文本
    print('=' * 70)
    print('光伏功率预测 - LLM 异常诊断演示(InternLM API)')
    print('=' * 70)
    print(f'API 地址: {API_URL}')
    print(f'模型: {MODEL}')
    print(f'API Key: {"已配置" if API_KEY else "未配置(将使用 mock)"}')

    # 场景1: 中度异常 + 阴雨天气
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

    # 场景2: 重度异常 + 天气正常
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

    print('\n' + '=' * 70)
    print('演示完成。')
    print('=' * 70)
