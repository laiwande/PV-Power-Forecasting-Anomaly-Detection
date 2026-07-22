# 光伏电站智能预测与异常诊断平台

基于 PatchTST 深度学习模型进行光伏功率预测,结合 Isolation Forest 异常检测与 InternLM 大模型智能诊断,通过 Streamlit + Plotly 提供交互式可视化。

## 系统架构图

```
┌──────────────────────────────────────────┐
│  Streamlit 可视化平台 (app.py)           │
│  历史功率 · 预测对比 · 异常标记 · 诊断报告│
└──────────────────┬───────────────────────┘
                   │
┌──────────────────▼───────────────────────┐
│  核心推理管线                             │
├────────────┬─────────────┬───────────────┤
│ PatchTST   │ Isolation   │ InternLM API  │
│ 功率预测   │ Forest      │ LLM 诊断      │
│ (predict)  │ 异常检测    │ (llm_analysis)│
├────────────┴─────────────┴───────────────┤
│  features.py 特征工程(太阳高度角/时间编码)│
├──────────────────────────────────────────┤
│  data/pv_dataset.csv (EDS-lab 数据集)    │
└──────────────────────────────────────────┘
```

## 技术栈

| 层级 | 技术 |
|---|---|
| 预测模型 | PatchTST (PyTorch 自实现, channel-independent) |
| 异常检测 | scikit-learn Isolation Forest |
| LLM 诊断 | InternLM (intern-latest, 云端 API) |
| 特征工程 | 太阳高度角/时间编码/滞后特征 (15+ 维) |
| 可视化 | Streamlit + Plotly |
| 主题 | 青花瓷器配色风格 |

## 快速开始

- 环境要求:Python ≥ 3.10
- 安装依赖:`pip install -r requirements.txt`
- 配置 .env(InternLM API Key)
- 训练模型:`python train.py`
- 启动平台:`streamlit run app.py`
- 模型评估:`python eval.py`

## 目录结构

```
PV-Power-Forecasting-Anomaly-Detection/
├── app.py              # Streamlit 可视化平台
├── train.py            # PatchTST 训练脚本
├── predict.py          # 推理脚本
├ anomaly.py            # Isolation Forest 异常检测
├── llm_analysis.py     # InternLM LLM 诊断
├── features.py         # 光伏特征工程
├── dispatch.py         # 运维决策模块
├── eval.py             # 模型评估对比
├── data/               # 数据集
│   ├── pv_dataset.csv
│   └── prepare_dataset.py
└── models/             # 训练产物
    ├── patchtst.pth
    └── scaler.pkl
```

## 许可证

MIT
