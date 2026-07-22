# Tasks

- [x] Task 1: 特征工程重构(train.py + predict.py)
  - [x] SubTask 1.1: 新建 `features.py` 模块,实现 `build_features(df)` 函数,构造太阳高度角(假设纬度 35°N)、hour_sin/hour_cos、is_daytime、shortwave_radiation 滞后 1/3/6 步、temperature 滞后 3 步等特征,返回扩展后的 DataFrame 与新 FEATURE_COLS 清单
  - [x] SubTask 1.2: 修改 `train.py`,调用 `build_features` 构造特征,FEATURE_COLS 改为从 features.py 导入,scaler.pkl 新增保存 `feature_cols` 字段,模型 n_features 改为新维度
  - [x] SubTask 1.3: 修改 `predict.py`,加载 scaler 时校验 feature_cols,自动调用 `build_features` 构造新特征后再送入模型,PatchTST 的 n_features 与训练保持一致

- [x] Task 2: 运维决策模块(dispatch.py)
  - [x] SubTask 2.1: 新建 `dispatch.py`,实现 `recommend_action(anomaly_result)` 函数,根据 anomaly_ratio 与严重程度输出 4 种动作之一(持续监测 0.96/现场巡检 0.86/降容运行 0.78/紧急停机 0.5),返回包含 action_name/coefficient/rationale/urgency 的 dict
  - [x] SubTask 2.2: 修改 `llm_analysis.py`,在 `_build_prompt` 中传入决策动作信息,prompt 新增"建议运维动作"段落,要求 LLM 结合决策给出具体执行建议
  - [x] SubTask 2.3: 修改 `app.py`,新增 `render_dispatch_card` 函数渲染运维决策卡片(动作名/系数/紧急度/说明),在异常检测后展示

- [x] Task 3: 评估模块(eval.py)
  - [x] SubTask 3.1: 新建 `eval.py`,实现 Persistence 基线(用最后 24 步历史值作为预测)与 LightGBM 基线(用 96 步特征平铺预测 24 步),在验证集上滑动窗口评估
  - [x] SubTask 3.2: 实现 MAE/RMSE/MAPE 指标计算,输出对比表到控制台,并用 matplotlib 生成 `eval_results.png`(柱状图对比三模型三指标)

- [x] Task 4: 文档与依赖(README.md + LICENSE + requirements.txt)
  - [x] SubTask 4.1: 新建 `LICENSE`(MIT),新建 `README.md`(项目简介、架构图、技术栈表格、快速开始、目录结构、评估说明)
  - [x] SubTask 4.2: 更新 `requirements.txt` 新增 `lightgbm>=4.0`、`matplotlib>=3.7`

# Task Dependencies
- Task 2 依赖 Task 1(决策模块需对接异常检测,异常检测依赖新特征模型)
- Task 3 依赖 Task 1(评估需用新特征模型的预测结果)
- Task 4 可与 Task 1/2/3 并行
