"""光伏电站智能预测与异常诊断平台 - Streamlit 可视化应用。

集成 PatchTST 功率预测、Isolation Forest 异常检测与 LLM 诊断,
通过 Plotly 交互式图表展示历史功率、预测功率、实际/预测对比、异常标记与诊断报告。
"""
import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# 确保能 import 同目录下的模块(predict / anomaly / llm_analysis)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from predict import predict_future
from anomaly import detect_anomalies
from llm_analysis import generate_diagnosis

# 采样与模型参数
SEQ_LEN = 96   # 历史窗口长度(96 小时)
PRED_LEN = 24  # 预测长度(24 小时)
DATA_PATH = os.path.join(BASE_DIR, 'data', 'pv_dataset.csv')


@st.cache_data
def load_data():
    """加载并预处理光伏数据集(带缓存)。"""
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def main():
    st.set_page_config(
        page_title='光伏电站智能预测与异常诊断平台',
        page_icon='☀️',
        layout='wide',
    )
    st.title('☀️ 光伏电站智能预测与异常诊断平台')
    st.markdown(
        '基于 **PatchTST** 模型的光伏功率预测与异常诊断可视化平台。'
        '在左侧选择历史窗口起点,即可查看未来 24 小时功率预测、实际/预测对比、'
        '异常点检测及 LLM 诊断报告。'
    )

    # ===== 数据加载 =====
    df = load_data()
    n = len(df)

    # ===== 侧边栏控制 =====
    st.sidebar.header('控制面板')

    # 预测起点:日期选择器(保证起点后至少有 96+24 步数据)
    min_date = df['timestamp'].iloc[0].date()
    max_date = df['timestamp'].iloc[n - SEQ_LEN - PRED_LEN].date()
    default_date = min_date + timedelta(days=180)
    if default_date > max_date:
        default_date = min_date

    selected_date = st.sidebar.date_input(
        '预测起点(历史窗口起始日)',
        min_value=min_date,
        max_value=max_date,
        value=default_date,
        help='选择一个日期作为 96 小时历史窗口的起始点,其后须保留至少 120 小时数据。',
    )

    # 异常检测敏感度
    contamination = st.sidebar.slider(
        '异常检测敏感度(contamination)',
        min_value=0.01,
        max_value=0.20,
        value=0.05,
        step=0.01,
        help='值越大,检测到的异常点越多。',
    )

    # 定位起始索引:所选日期的第一个时间点
    selected_ts = pd.Timestamp(selected_date)
    mask = df['timestamp'] >= selected_ts
    start_idx = int(df.index[mask][0]) if mask.any() else 0

    # 显示所选窗口信息
    hist_start = df['timestamp'].iloc[start_idx]
    hist_end = df['timestamp'].iloc[start_idx + SEQ_LEN - 1]
    pred_start = df['timestamp'].iloc[start_idx + SEQ_LEN]
    pred_end = df['timestamp'].iloc[start_idx + SEQ_LEN + PRED_LEN - 1]
    st.sidebar.markdown('---')
    st.sidebar.markdown(f'**历史窗口**\n\n{hist_start} ~ {hist_end}')
    st.sidebar.markdown(f'**预测时段**\n\n{pred_start} ~ {pred_end}')

    # 预测按钮
    run_btn = st.sidebar.button('🚀 开始预测与分析', type='primary')

    # ===== 未点击时:展示数据概览 =====
    if not run_btn:
        st.subheader('数据集概览')
        c1, c2, c3 = st.columns(3)
        c1.metric('数据总量', f'{n} 条')
        c2.metric('起始时间', str(df['timestamp'].iloc[0]))
        c3.metric('结束时间', str(df['timestamp'].iloc[-1]))
        st.caption('在左侧选择预测起点并点击「开始预测与分析」按钮启动预测流程。')
        return

    # ===== 点击按钮:执行预测分析流程 =====
    # 1. 取历史 96 步 + 未来 24 步实际数据
    history = df.iloc[start_idx: start_idx + SEQ_LEN].reset_index(drop=True)
    future = df.iloc[start_idx + SEQ_LEN: start_idx + SEQ_LEN + PRED_LEN].reset_index(drop=True)
    actual = future['y'].values.astype(float)
    future_ts = future['timestamp'].values
    hist_ts = history['timestamp'].values
    hist_y = history['y'].values.astype(float)

    # 2. 调用 PatchTST 预测未来 24 步
    with st.spinner('正在运行 PatchTST 模型预测未来 24 小时...'):
        try:
            predicted = predict_future(history)
        except Exception as e:
            st.error(f'预测失败: {e}')
            st.error(
                '请确认已运行 train.py 生成模型文件:\n'
                '- models/scaler.pkl\n'
                '- models/patchtst.pth'
            )
            return

    # 3. 异常检测(实际 vs 预测)
    with st.spinner('正在执行异常检测(Isolation Forest)...'):
        anomaly_result = detect_anomalies(actual, predicted, contamination=contamination)

    # 4. 取对应时段天气数据
    weather_df = future[['temperature_2m', 'cloud_cover', 'shortwave_radiation', 'relative_humidity_2m']]

    # 5. LLM 诊断
    with st.spinner('正在生成 LLM 诊断报告...'):
        diagnosis = generate_diagnosis(anomaly_result, weather_df)

    # ===== 展示模块 =====

    # 模块 1: 功率预测曲线(历史 96 步 + 预测 24 步)
    st.subheader('1. 功率预测曲线(历史 96 步 + 预测 24 步)')
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=hist_ts, y=hist_y, mode='lines', name='历史功率',
        line=dict(color='royalblue', width=2),
    ))
    fig1.add_trace(go.Scatter(
        x=future_ts, y=predicted, mode='lines', name='预测功率',
        line=dict(color='orange', width=2, dash='dash'),
    ))
    fig1.update_layout(
        xaxis_title='时间', yaxis_title='功率',
        hovermode='x unified', height=420,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )
    st.plotly_chart(fig1, use_container_width=True)

    # 模块 2: 实际功率 vs 预测功率对比
    st.subheader('2. 实际功率 vs 预测功率对比')
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=future_ts, y=actual, mode='lines+markers', name='实际功率',
        line=dict(color='green', width=2),
        marker=dict(size=6),
    ))
    fig2.add_trace(go.Scatter(
        x=future_ts, y=predicted, mode='lines+markers', name='预测功率',
        line=dict(color='orange', width=2, dash='dash'),
        marker=dict(size=6),
    ))
    fig2.update_layout(
        xaxis_title='时间', yaxis_title='功率',
        hovermode='x unified', height=420,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # 模块 3: 异常点标记与统计
    st.subheader('3. 异常点标记与统计')
    anomaly_indices = anomaly_result['anomaly_indices']
    n_anom = anomaly_result['n_anomalies']
    anom_ratio = anomaly_result['anomaly_ratio']

    col_a, col_b, col_c = st.columns(3)
    col_a.metric('异常点数量', f'{n_anom}')
    col_b.metric('异常点占比', f'{anom_ratio * 100:.2f}%')
    col_c.metric('检测敏感度', f'{contamination:.2f}')

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=future_ts, y=actual, mode='lines+markers', name='实际功率',
        line=dict(color='green', width=2),
        marker=dict(size=6),
    ))
    fig3.add_trace(go.Scatter(
        x=future_ts, y=predicted, mode='lines+markers', name='预测功率',
        line=dict(color='orange', width=2, dash='dash'),
        marker=dict(size=6),
    ))
    if n_anom > 0:
        anom_ts = future_ts[anomaly_indices]
        anom_actual = actual[anomaly_indices]
        fig3.add_trace(go.Scatter(
            x=anom_ts, y=anom_actual, mode='markers', name='异常点',
            marker=dict(color='red', size=14, symbol='x', line=dict(width=2)),
        ))
    fig3.update_layout(
        xaxis_title='时间', yaxis_title='功率',
        hovermode='x unified', height=420,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # 模块 4: LLM 诊断结果
    st.subheader('4. LLM 诊断结果')
    st.info(diagnosis.replace('\n', '\n\n'))


if __name__ == '__main__':
    main()
