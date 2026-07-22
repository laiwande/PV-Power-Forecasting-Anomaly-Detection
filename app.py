"""光伏电站智能预测与异常诊断平台 - Streamlit 可视化应用。

集成 PatchTST 功率预测、Isolation Forest 异常检测与 LLM 诊断,
通过 Plotly 交互式图表展示历史功率、预测功率、实际/预测对比、异常标记与诊断报告。

UI 设计:深色科技感光伏运维仪表盘,太阳能金/琥珀色主题。
"""
import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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

# ===== 主题配色(太阳能科技风)=====
THEME = {
    'bg': '#0a0e1a',           # 深空蓝背景
    'card_bg': '#141b2d',      # 卡片背景
    'card_border': '#1f2940',  # 卡片边框
    'primary': '#f59e0b',      # 太阳能金
    'primary_light': '#fbbf24', # 浅琥珀
    'secondary': '#3b82f6',    # 科技蓝
    'success': '#10b981',      # 翠绿(实际功率)
    'danger': '#ef4444',       # 警示红(异常点)
    'warning': '#f97316',      # 橙色(预测功率)
    'text': '#e2e8f0',         # 主文字
    'text_muted': '#94a3b8',   # 次要文字
    'grid': '#1e293b',         # 网格线
}


# ===== 自定义 CSS 注入 =====
def inject_custom_css():
    """注入自定义 CSS,实现深色科技感光伏运维仪表盘风格。"""
    st.markdown("""
    <style>
    /* ===== 全局背景与字体 ===== */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0f1729 100%);
        color: #e2e8f0;
    }
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background:
            radial-gradient(circle at 20% 10%, rgba(245, 158, 11, 0.06) 0%, transparent 40%),
            radial-gradient(circle at 80% 80%, rgba(59, 130, 246, 0.05) 0%, transparent 40%);
        pointer-events: none;
        z-index: 0;
    }
    /* 确保所有 Streamlit 内容在伪元素之上,可正常交互 */
    .stApp > * { position: relative; z-index: 1; }
    section[data-testid="stSidebar"] { z-index: 2; }

    /* ===== 顶部标题栏 ===== */
    .main-header {
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.12) 0%, rgba(59, 130, 246, 0.08) 100%);
        border-left: 4px solid #f59e0b;
        border-radius: 0 12px 12px 0;
        padding: 20px 28px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .main-header::after {
        content: '';
        position: absolute;
        top: 0; right: 0;
        width: 200px; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(245, 158, 11, 0.04));
    }
    .main-header h1 {
        font-size: 26px;
        font-weight: 700;
        margin: 0 0 6px 0;
        color: #fbbf24;
        letter-spacing: 0.5px;
    }
    .main-header p {
        font-size: 13px;
        margin: 0;
        color: #94a3b8;
        line-height: 1.6;
    }

    /* ===== 统计卡片 ===== */
    .metric-card {
        background: linear-gradient(135deg, #141b2d 0%, #1a2238 100%);
        border: 1px solid #1f2940;
        border-radius: 12px;
        padding: 20px 24px;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 3px; height: 100%;
        background: #f59e0b;
    }
    .metric-card.success::before { background: #10b981; }
    .metric-card.warning::before { background: #f97316; }
    .metric-card.danger::before { background: #ef4444; }
    .metric-card.info::before { background: #3b82f6; }
    .metric-card .label {
        font-size: 12px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    .metric-card .value {
        font-size: 28px;
        font-weight: 700;
        color: #fbbf24;
    }
    .metric-card.success .value { color: #10b981; }
    .metric-card.warning .value { color: #f97316; }
    .metric-card.danger .value { color: #ef4444; }
    .metric-card.info .value { color: #60a5fa; }
    .metric-card .unit {
        font-size: 14px;
        font-weight: 400;
        color: #94a3b8;
        margin-left: 4px;
    }

    /* ===== 分区标题 ===== */
    .section-title {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 28px 0 16px 0;
        padding-bottom: 12px;
        border-bottom: 1px solid #1f2940;
    }
    .section-title .icon {
        font-size: 20px;
    }
    .section-title .text {
        font-size: 18px;
        font-weight: 600;
        color: #e2e8f0;
    }
    .section-title .badge {
        font-size: 11px;
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        padding: 2px 10px;
        border-radius: 10px;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    /* ===== 诊断报告卡片 ===== */
    .diagnosis-card {
        background: linear-gradient(135deg, #141b2d 0%, #1a2238 100%);
        border: 1px solid #1f2940;
        border-left: 4px solid #f59e0b;
        border-radius: 12px;
        padding: 24px 28px;
        white-space: pre-wrap;
        line-height: 1.8;
        font-size: 14px;
        color: #cbd5e1;
        position: relative;
    }
    .diagnosis-card .header {
        font-size: 15px;
        font-weight: 600;
        color: #fbbf24;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid #1f2940;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ===== Streamlit 组件覆盖样式 ===== */
    .stButton > button {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: #0a0e1a;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);
        transform: translateY(-1px);
    }

    /* 侧边栏 */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1729 0%, #0a0e1a 100%);
        border-right: 1px solid #1f2940;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #fbbf24;
    }

    /* 标签与文字颜色 */
    .stMarkdown, .stText { color: #e2e8f0; }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #fbbf24; }

    /* Plotly 图表背景透明 */
    .stPlotlyChart { background: transparent; }

    /* 隐藏默认 Streamlit 元素 */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* 空状态提示 */
    .empty-state {
        text-align: center;
        padding: 80px 20px;
        color: #64748b;
    }
    .empty-state .icon {
        font-size: 64px;
        margin-bottom: 20px;
        opacity: 0.4;
    }
    .empty-state .title {
        font-size: 20px;
        color: #94a3b8;
        margin-bottom: 8px;
    }
    .empty-state .desc {
        font-size: 14px;
        color: #64748b;
    }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    """渲染顶部标题栏。"""
    st.markdown("""
    <div class="main-header">
        <h1>☀️ 光伏电站智能预测与异常诊断平台</h1>
        <p>基于 PatchTST 深度学习模型的光伏功率预测 · Isolation Forest 异常检测 · InternLM 大模型智能诊断</p>
    </div>
    """, unsafe_allow_html=True)


def render_metric_card(label, value, unit='', card_type='primary'):
    """渲染统计卡片。"""
    st.markdown(f"""
    <div class="metric-card {card_type}">
        <div class="label">{label}</div>
        <div class="value">{value}<span class="unit">{unit}</span></div>
    </div>
    """, unsafe_allow_html=True)


def render_section_title(icon, text, badge=''):
    """渲染分区标题。"""
    badge_html = f'<span class="badge">{badge}</span>' if badge else ''
    st.markdown(f"""
    <div class="section-title">
        <span class="icon">{icon}</span>
        <span class="text">{text}</span>
        {badge_html}
    </div>
    """, unsafe_allow_html=True)


def render_empty_state(icon, title, desc):
    """渲染空状态。"""
    st.markdown(f"""
    <div class="empty-state">
        <div class="icon">{icon}</div>
        <div class="title">{title}</div>
        <div class="desc">{desc}</div>
    </div>
    """, unsafe_allow_html=True)


def render_diagnosis_card(content):
    """渲染 LLM 诊断报告卡片。"""
    st.markdown(f"""
    <div class="diagnosis-card">
        <div class="header">🤖 InternLM 智能诊断报告</div>
        {content}
    </div>
    """, unsafe_allow_html=True)


@st.cache_data
def load_data():
    """加载并预处理光伏数据集(带缓存)。"""
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def build_dark_plotly_layout(title='', y_title='功率 (kW)'):
    """构建深色主题的 Plotly 布局配置。"""
    return dict(
        title=dict(text=title, font=dict(color=THEME['text'], size=15), x=0.02),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(20,27,45,0.6)',
        font=dict(color=THEME['text'], family='system-ui'),
        xaxis=dict(
            title='时间',
            gridcolor=THEME['grid'],
            zerolinecolor=THEME['grid'],
            color=THEME['text_muted'],
        ),
        yaxis=dict(
            title=y_title,
            gridcolor=THEME['grid'],
            zerolinecolor=THEME['grid'],
            color=THEME['text_muted'],
        ),
        hovermode='x unified',
        hoverlabel=dict(bgcolor=THEME['card_bg'], font_color=THEME['text']),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
            font=dict(color=THEME['text_muted']),
            bgcolor='rgba(0,0,0,0)',
        ),
        margin=dict(l=20, r=20, t=50, b=20),
    )


def main():
    st.set_page_config(
        page_title='光伏电站智能预测与异常诊断平台',
        page_icon='☀️',
        layout='wide',
    )

    # 注入自定义样式
    inject_custom_css()
    render_header()

    # ===== 数据加载 =====
    df = load_data()
    n = len(df)

    # ===== 侧边栏控制 =====
    with st.sidebar:
        st.markdown('### ⚙️ 控制面板')
        st.markdown('---')

        # 预测起点:日期选择器
        min_date = df['timestamp'].iloc[0].date()
        max_date = df['timestamp'].iloc[n - SEQ_LEN - PRED_LEN].date()
        default_date = min_date + timedelta(days=180)
        if default_date > max_date:
            default_date = min_date

        st.markdown('#### 📅 预测起点')
        selected_date = st.date_input(
            '历史窗口起始日',
            min_value=min_date,
            max_value=max_date,
            value=default_date,
            help='选择一个日期作为 96 小时历史窗口的起始点,其后须保留至少 120 小时数据。',
            label_visibility='collapsed',
        )

        st.markdown('#### 🎯 异常检测敏感度')
        contamination = st.slider(
            'contamination',
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01,
            help='值越大,检测到的异常点越多。',
            label_visibility='collapsed',
        )

        # 定位起始索引
        selected_ts = pd.Timestamp(selected_date)
        mask = df['timestamp'] >= selected_ts
        start_idx = int(df.index[mask][0]) if mask.any() else 0

        # 显示所选窗口信息
        hist_start = df['timestamp'].iloc[start_idx]
        hist_end = df['timestamp'].iloc[start_idx + SEQ_LEN - 1]
        pred_start = df['timestamp'].iloc[start_idx + SEQ_LEN]
        pred_end = df['timestamp'].iloc[start_idx + SEQ_LEN + PRED_LEN - 1]

        st.markdown('---')
        st.markdown('#### 📍 窗口信息')
        st.markdown(
            f"<div style='background:#141b2d; padding:12px 16px; border-radius:8px; "
            f"border-left:3px solid #3b82f6; margin-bottom:8px; font-size:13px;'>"
            f"<div style='color:#94a3b8; font-size:11px; margin-bottom:4px;'>历史窗口</div>"
            f"<div style='color:#60a5fa;'>{hist_start.strftime('%Y-%m-%d %H:%M')}<br>"
            f"~ {hist_end.strftime('%Y-%m-%d %H:%M')}</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='background:#141b2d; padding:12px 16px; border-radius:8px; "
            f"border-left:3px solid #f59e0b; margin-bottom:8px; font-size:13px;'>"
            f"<div style='color:#94a3b8; font-size:11px; margin-bottom:4px;'>预测时段</div>"
            f"<div style='color:#fbbf24;'>{pred_start.strftime('%Y-%m-%d %H:%M')}<br>"
            f"~ {pred_end.strftime('%Y-%m-%d %H:%M')}</div></div>",
            unsafe_allow_html=True,
        )

        st.markdown('---')
        run_btn = st.button('🚀 开始预测与分析', type='primary')

    # ===== 未点击时:展示数据概览 =====
    if not run_btn:
        render_section_title('📊', '数据集概览', '就绪')

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_metric_card('数据总量', f'{n:,}', '条', 'info')
        with col2:
            render_metric_card('起始时间', str(df['timestamp'].iloc[0])[:10], '', 'success')
        with col3:
            render_metric_card('结束时间', str(df['timestamp'].iloc[-1])[:10], '', 'success')
        with col4:
            render_metric_card('采样间隔', '1', '小时', 'warning')

        render_empty_state(
            '☀️',
            '等待启动预测分析',
            '请在左侧控制面板选择预测起点,点击「开始预测与分析」按钮启动智能诊断流程',
        )
        return

    # ===== 点击按钮:执行预测分析流程 =====
    # 1. 取历史 96 步 + 未来 24 步实际数据
    history = df.iloc[start_idx: start_idx + SEQ_LEN].reset_index(drop=True)
    future = df.iloc[start_idx + SEQ_LEN: start_idx + SEQ_LEN + PRED_LEN].reset_index(drop=True)
    actual = future['y'].values.astype(float)
    future_ts = future['timestamp'].values
    hist_ts = history['timestamp'].values
    hist_y = history['y'].values.astype(float)

    # 2. 调用 PatchTST 预测
    with st.spinner('⚡ PatchTST 模型预测中...'):
        try:
            predicted = predict_future(history)
        except Exception as e:
            st.error(f'预测失败: {e}')
            st.error('请确认已运行 train.py 生成模型文件:models/scaler.pkl 和 models/patchtst.pth')
            return

    # 3. 异常检测
    with st.spinner('🔍 Isolation Forest 异常检测中...'):
        anomaly_result = detect_anomalies(actual, predicted, contamination=contamination)

    # 4. 天气数据
    weather_df = future[['temperature_2m', 'cloud_cover', 'shortwave_radiation', 'relative_humidity_2m']]

    # 5. LLM 诊断
    with st.spinner('🤖 InternLM 智能诊断生成中...'):
        diagnosis = generate_diagnosis(anomaly_result, weather_df)

    # ===== 关键指标概览 =====
    anomaly_indices = anomaly_result['anomaly_indices']
    n_anom = anomaly_result['n_anomalies']
    anom_ratio = anomaly_result['anomaly_ratio']
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    mae = float(np.mean(np.abs(actual - predicted)))

    render_section_title('📈', '关键指标', '实时')
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_metric_card('异常点数量', f'{n_anom}', '个', 'danger' if n_anom > 0 else 'success')
    with col2:
        render_metric_card('异常占比', f'{anom_ratio * 100:.2f}', '%', 'danger' if anom_ratio > 0.1 else 'warning')
    with col3:
        render_metric_card('RMSE', f'{rmse:.4f}', '', 'info')
    with col4:
        render_metric_card('MAE', f'{mae:.4f}', '', 'info')
    with col5:
        render_metric_card('检测敏感度', f'{contamination:.2f}', '', 'warning')

    # ===== 模块 1: 功率预测曲线 =====
    render_section_title('🔋', '功率预测曲线', '历史 96h + 预测 24h')
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=hist_ts, y=hist_y, mode='lines', name='历史功率',
        line=dict(color=THEME['secondary'], width=2.5),
        fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.08)',
    ))
    fig1.add_trace(go.Scatter(
        x=[hist_ts[-1], future_ts[0]], y=[hist_y[-1], predicted[0]],
        mode='lines', name='衔接',
        line=dict(color=THEME['text_muted'], width=1, dash='dot'),
        showlegend=False,
    ))
    fig1.add_trace(go.Scatter(
        x=future_ts, y=predicted, mode='lines+markers', name='预测功率',
        line=dict(color=THEME['warning'], width=2.5, dash='dash'),
        marker=dict(size=5, color=THEME['warning']),
    ))
    # 预测区域背景
    fig1.add_vrect(
        x0=future_ts[0], x1=future_ts[-1],
        fillcolor='rgba(245, 158, 11, 0.06)', layer='below', line_width=0,
    )
    fig1.update_layout(build_dark_plotly_layout('功率预测曲线'), height=420)
    st.plotly_chart(fig1, use_container_width=True)

    # ===== 模块 2 & 3: 并排展示对比图 + 异常标记图 =====
    col_left, col_right = st.columns(2)

    with col_left:
        render_section_title('⚖️', '实际 vs 预测', '对比')
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=future_ts, y=actual, mode='lines+markers', name='实际功率',
            line=dict(color=THEME['success'], width=2.5),
            marker=dict(size=6),
            fill='tozeroy', fillcolor='rgba(16, 185, 129, 0.08)',
        ))
        fig2.add_trace(go.Scatter(
            x=future_ts, y=predicted, mode='lines+markers', name='预测功率',
            line=dict(color=THEME['warning'], width=2.5, dash='dash'),
            marker=dict(size=6),
        ))
        fig2.update_layout(build_dark_plotly_layout('实际 vs 预测对比'), height=400)
        st.plotly_chart(fig2, use_container_width=True)

    with col_right:
        render_section_title('🚨', '异常点标记', f'{n_anom} 个异常')
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=future_ts, y=actual, mode='lines+markers', name='实际功率',
            line=dict(color=THEME['success'], width=2),
            marker=dict(size=5),
        ))
        fig3.add_trace(go.Scatter(
            x=future_ts, y=predicted, mode='lines', name='预测功率',
            line=dict(color=THEME['warning'], width=2, dash='dash'),
        ))
        if n_anom > 0:
            anom_ts = future_ts[anomaly_indices]
            anom_actual = actual[anomaly_indices]
            fig3.add_trace(go.Scatter(
                x=anom_ts, y=anom_actual, mode='markers', name='异常点',
                marker=dict(
                    color=THEME['danger'], size=16, symbol='x',
                    line=dict(width=2, color=THEME['danger']),
                ),
            ))
        fig3.update_layout(build_dark_plotly_layout('异常点检测'), height=400)
        st.plotly_chart(fig3, use_container_width=True)

    # ===== 模块 4: 天气环境数据 =====
    render_section_title('🌤️', '天气环境数据', '预测时段')
    col_w1, col_w2, col_w3, col_w4 = st.columns(4)
    with col_w1:
        render_metric_card('温度', f'{weather_df["temperature_2m"].mean():.1f}', '°C', 'info')
    with col_w2:
        render_metric_card('云量', f'{weather_df["cloud_cover"].mean():.1f}', '%', 'warning')
    with col_w3:
        render_metric_card('短波辐射', f'{weather_df["shortwave_radiation"].mean():.1f}', 'W/m²', 'primary')
    with col_w4:
        render_metric_card('相对湿度', f'{weather_df["relative_humidity_2m"].mean():.1f}', '%', 'info')

    # ===== 模块 5: LLM 诊断报告 =====
    render_section_title('🤖', 'LLM 智能诊断', 'InternLM')
    # 将诊断文本格式化为 HTML(保留换行,转义特殊字符)
    diagnosis_html = diagnosis.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    diagnosis_html = diagnosis_html.replace('\n', '<br>')
    render_diagnosis_card(diagnosis_html)


if __name__ == '__main__':
    main()
