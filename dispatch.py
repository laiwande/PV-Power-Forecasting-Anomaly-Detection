"""光伏电站运维决策模块。

根据异常检测结果(detect_anomalies 返回的 dict)推荐运维动作,
输出动作名称、动作系数、决策依据与紧急程度,供可视化与 LLM 诊断共用。

决策阈值基于异常占比(anomaly_ratio):
    - 0 或 < 0.10  → 持续监测(系数 0.96, 低)
    - 0.10 ~ 0.20  → 现场巡检(系数 0.86, 中)
    - 0.20 ~ 0.35  → 降容运行(系数 0.78, 高)
    - >= 0.35      → 紧急停机(系数 0.50, 紧急)
"""


def recommend_action(anomaly_result):
    """根据异常检测结果推荐运维动作。

    Args:
        anomaly_result: detect_anomalies 返回的 dict,含 n_anomalies/anomaly_ratio/scores 等。

    Returns:
        dict,含:
            - action_name: 动作名称(持续监测/现场巡检/降容运行/紧急停机)
            - coefficient: 动作系数(0.96/0.86/0.78/0.5)
            - rationale: 决策依据说明
            - urgency: 紧急程度(低/中/高/紧急)
    """
    # 兼容 None 或空输入
    if not anomaly_result:
        ratio = 0.0
        n_anomalies = 0
    else:
        n_anomalies = int(anomaly_result.get('n_anomalies', 0) or 0)
        # anomaly_ratio 优先,缺失时根据 labels 推算
        ratio = anomaly_result.get('anomaly_ratio', None)
        if ratio is None:
            labels = anomaly_result.get('labels')
            if labels is not None and len(labels) > 0:
                import numpy as _np
                ratio = float(_np.sum(_np.asarray(labels) == -1) / len(labels))
            else:
                ratio = 0.0
        ratio = float(ratio or 0.0)

    # 决策规则
    if n_anomalies == 0 or ratio == 0 or ratio < 0.1:
        return {
            'action_name': '持续监测',
            'coefficient': 0.96,
            'rationale': '异常占比低,设备运行正常,建议保持常规监测频率',
            'urgency': '低',
        }
    if ratio < 0.2:
        return {
            'action_name': '现场巡检',
            'coefficient': 0.86,
            'rationale': '异常占比上升,存在潜在风险,建议安排现场巡检排查设备状态',
            'urgency': '中',
        }
    if ratio < 0.35:
        return {
            'action_name': '降容运行',
            'coefficient': 0.78,
            'rationale': '异常占比较高,设备可能出现性能衰减,建议降容运行以保护设备',
            'urgency': '高',
        }
    return {
        'action_name': '紧急停机',
        'coefficient': 0.5,
        'rationale': '异常占比严重,设备存在高风险,建议立即停机检修避免事故',
        'urgency': '紧急',
    }


if __name__ == '__main__':
    # 命令行演示: 不同异常占比下的运维决策
    print('=' * 70)
    print('光伏电站运维决策演示')
    print('=' * 70)

    demo_cases = [
        {'n_anomalies': 0, 'anomaly_ratio': 0.0, 'labels': [1, 1, 1]},
        {'n_anomalies': 1, 'anomaly_ratio': 0.05, 'labels': [1, 1, -1]},
        {'n_anomalies': 3, 'anomaly_ratio': 0.15, 'labels': [1, -1, -1, -1]},
        {'n_anomalies': 6, 'anomaly_ratio': 0.25, 'labels': [1, -1, -1, -1]},
        {'n_anomalies': 10, 'anomaly_ratio': 0.40, 'labels': [-1, -1, -1, -1]},
    ]

    for i, case in enumerate(demo_cases, 1):
        result = recommend_action(case)
        print(f'\n[场景 {i}] anomaly_ratio={case["anomaly_ratio"]:.2f}, n_anomalies={case["n_anomalies"]}')
        print(f'  动作名称: {result["action_name"]}')
        print(f'  动作系数: {result["coefficient"]}')
        print(f'  紧急程度: {result["urgency"]}')
        print(f'  决策依据: {result["rationale"]}')

    print('\n' + '=' * 70)
    print('演示完成。')
    print('=' * 70)
