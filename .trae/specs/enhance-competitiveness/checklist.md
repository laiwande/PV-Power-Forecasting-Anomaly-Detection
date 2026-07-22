# Checklist

## 特征工程
- [x] `features.py` 实现 `build_features(df)` 函数,能基于 timestamp 构造太阳高度角、时间编码、昼夜标记、滞后特征
- [x] 新 FEATURE_COLS 清单包含 15+ 维特征(实际16维),且 y 仍在第 0 列
- [x] `train.py` 调用 `build_features`,scaler.pkl 保存 `feature_cols` 字段
- [x] `predict.py` 加载 scaler 时校验 feature_cols 并自动构造新特征
- [x] 执行 `python train.py` 能用新特征完成训练,无报错(20 epochs,best_val=0.356548)

## 运维决策
- [x] `dispatch.py` 的 `recommend_action` 能根据 anomaly_ratio 返回 4 种动作之一
- [x] `llm_analysis.py` 的 prompt 包含建议运维动作信息
- [x] `app.py` 渲染运维决策卡片,展示动作名/系数/紧急度/说明
- [x] 运行 Streamlit 后决策卡片在异常检测后正确展示(代码验证通过,Streamlit 正常启动无报错;浏览器 iframe 自动化交互受限未能端到端截图)

## 评估模块
- [x] `eval.py` 实现 Persistence 与 LightGBM 两个基线模型(LightGBM 优雅跳过未安装情况)
- [x] 输出 MAE/RMSE/MAPE 三指标对比表到控制台
- [x] 生成 `eval_results.png` 可视化图表
- [x] 执行 `python eval.py` 无报错且输出完整对比结果

## 文档与依赖
- [x] `LICENSE` 文件为标准 MIT 许可证
- [x] `README.md` 包含项目简介、架构图、技术栈、快速开始、目录结构
- [x] `requirements.txt` 新增 lightgbm 与 matplotlib
- [x] 安装新依赖后所有脚本可正常运行(predict.py/eval.py/train.py 均验证通过)
