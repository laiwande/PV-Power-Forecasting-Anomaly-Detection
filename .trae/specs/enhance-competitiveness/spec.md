# 光伏预测平台竞争力增强 Spec

## Why
对比 `VegetableoVoBird/wind-power-forecasting-rl-dispatch` 后,本项目在工程完整度、特征工程深度、业务决策闭环三方面存在明显差距。本 spec 通过补评估模块与文档、深化特征工程、构建运维决策闭环,全面提升项目竞争力与学术严谨性。

## What Changes
- **新增评估模块 `eval.py`**:独立对比 PatchTST 与 Persistence、LightGBM 基线模型,输出 MAE/RMSE/MAE 指标对比表与可视化图表
- **新增 `README.md` 与 `LICENSE`**:补全项目文档(架构图、快速开始、API说明)与 MIT 许可证
- **深化特征工程**:在 `train.py` 与 `predict.py` 中新增光伏物理特征(太阳高度角、辐照度滞后、昼夜标记、时间编码),特征从 5 维扩展到 15+ 维
- **新增运维决策模块 `dispatch.py`**:基于异常检测结果输出 4 种运维建议动作(持续监测/现场巡检/降容运行/紧急停机),并集成到 `app.py` 可视化
- **重构 `train.py`**:支持新旧两套特征列配置,保持向后兼容;scaler 保存新增特征列清单

## Impact
- Affected code: `train.py`(特征工程重构)、`predict.py`(特征工程同步)、`anomaly.py`(对接决策模块)、`app.py`(展示运维建议)、`requirements.txt`(新增 lightgbm)、`llm_analysis.py`(决策动作传入诊断 prompt)
- 新增文件: `eval.py`、`dispatch.py`、`README.md`、`LICENSE`
- **BREAKING**: `FEATURE_COLS` 从固定 5 列扩展为可配置,已训练的 `models/patchtst.pth` 需用新特征重新训练

## ADDED Requirements

### Requirement: 模型评估对比
系统 SHALL 提供独立评估脚本 `eval.py`,对比 PatchTST 与 Persistence、LightGBM 基线模型在验证集上的预测精度。

#### Scenario: 评估运行
- **WHEN** 用户执行 `python eval.py`
- **THEN** 系统加载验证集,分别用三种模型预测,输出 MAE/RMSE/MAPE 指标对比表
- **AND** 生成 `eval_results.png` 可视化图表保存到项目根目录

### Requirement: 光伏物理特征工程
系统 SHALL 在数据预处理阶段自动构造光伏领域特征,包括太阳高度角、辐照度滞后项、昼夜标记、时间编码。

#### Scenario: 特征构造
- **WHEN** 训练或推理时加载 `pv_dataset.csv`
- **THEN** 系统基于 timestamp 计算太阳高度角(假设中纬度 35°N)、hour_sin/hour_cos 时间编码、is_daytime 昼夜标记、shortwave_radiation 滞后 1/3/6 步特征
- **AND** 特征列从 5 维扩展到 15+ 维

### Requirement: 运维决策建议
系统 SHALL 基于异常检测结果输出 4 种运维建议动作,并给出动作系数与执行说明。

#### Scenario: 决策输出
- **WHEN** 异常检测完成
- **THEN** 系统根据异常占比与严重程度输出建议动作(持续监测/现场巡检/降容运行/紧急停机)
- **AND** 在 `app.py` 可视化界面以卡片形式展示决策结果
- **AND** 决策动作信息传入 LLM 诊断 prompt,生成针对性运维建议

### Requirement: 项目文档
系统 SHALL 提供 `README.md` 项目说明与 `LICENSE` 开源许可证。

#### Scenario: 文档完整
- **WHEN** 用户访问 GitHub 仓库
- **THEN** 显示项目简介、架构图、技术栈、快速开始、API 说明、目录结构
- **AND** 包含 MIT License 文件

## MODIFIED Requirements

### Requirement: 训练流程
训练脚本 `train.py` SHALL 支持扩展后的特征列,并在 scaler 中持久化特征列清单,供推理时校验。

### Requirement: 推理流程
推理脚本 `predict.py` SHALL 加载与训练一致的特征列清单,自动构造新增特征后再送入模型。

### Requirement: 可视化平台
`app.py` SHALL 新增"运维决策"分区,展示决策动作卡片,并在诊断报告分区体现决策建议。
