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
from dispatch import recommend_action

# 采样与模型参数
SEQ_LEN = 96   # 历史窗口长度(96 小时)
PRED_LEN = 24  # 预测长度(24 小时)
DATA_PATH = os.path.join(BASE_DIR, 'data', 'pv_dataset.csv')

# ===== 主题配色(黑白极简风)=====
# 纯白底 · 纯黑字 · 灰阶层次 · 红色仅用于异常焦点
THEME = {
    'bg': '#F5F5F5',           # 浅灰(页面背景)
    'card_bg': '#FFFFFF',      # 纯白(卡片)
    'card_border': '#E0E0E0',  # 浅灰(细边框)
    'primary': '#111111',      # 纯黑(主色/CTA)
    'primary_light': '#444444', # 深灰
    'secondary': '#888888',    # 中灰(历史功率)
    'success': '#333333',      # 深灰(实际功率)
    'danger': '#CC0000',       # 红色(异常点/视觉焦点)
    'warning': '#666666',      # 中灰(预测功率)
    'text': '#222222',         # 近黑(主文字)
    'text_muted': '#999999',   # 浅灰(次要文字)
    'grid': 'rgba(0, 0, 0, 0.08)',  # 黑色网格(半透)
}


# ===== 自定义 CSS 注入 =====
def inject_custom_css():
    """注入自定义 CSS,实现黑白极简风格。"""
    st.markdown("""
    <style>
    /* ===== 全局背景与字体 ===== */
    .stApp {
        background: #F5F5F5;
        color: #222222;
        font-family: system-ui, -apple-system, "Segoe UI", "Noto Sans SC", sans-serif;
    }

    /* ===== 顶部标题栏 ===== */
    .main-header {
        border-bottom: 2px solid #111111;
        padding: 10px 0 22px 0;
        margin-bottom: 28px;
        position: relative;
    }
    .main-header::after {
        content: "";
        position: absolute;
        left: 0; bottom: -2px;
        width: 60px; height: 2px;
        background: #CC0000;
    }
    .main-header h1 {
        font-size: 30px;
        font-weight: 700;
        margin: 0 0 8px 0;
        color: #111111;
        letter-spacing: 0.5px;
        line-height: 1.2;
    }
    .main-header p {
        font-size: 13px;
        margin: 0;
        color: #999999;
        line-height: 1.6;
    }

    /* ===== 统计卡片(白底 + 黑色顶线) ===== */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-top: 3px solid #111111;
        border-radius: 4px;
        padding: 20px 24px;
        transition: all 0.25s ease;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
    }
    .metric-card:hover {
        border-top-color: #CC0000;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.10);
        transform: translateY(-2px);
    }
    .metric-card .label {
        font-size: 11px;
        color: #999999;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 8px;
    }
    .metric-card .value {
        font-size: 28px;
        font-weight: 700;
        color: #111111;
    }
    .metric-card.success .value { color: #333333; }
    .metric-card.warning .value { color: #666666; }
    .metric-card.danger .value { color: #CC0000; }
    .metric-card.info .value { color: #444444; }
    .metric-card.primary .value { color: #111111; }
    .metric-card .unit {
        font-size: 14px;
        font-weight: 400;
        color: #999999;
        margin-left: 4px;
    }

    /* ===== 分区标题 ===== */
    .section-title {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 34px 0 16px 0;
        padding-bottom: 12px;
        border-bottom: 1px solid #E0E0E0;
        position: relative;
    }
    .section-title::after {
        content: "";
        position: absolute;
        left: 0; bottom: -1px;
        width: 48px; height: 2px;
        background: #111111;
    }
    .section-title .icon {
        font-size: 18px;
    }
    .section-title .text {
        font-size: 19px;
        font-weight: 700;
        color: #111111;
        letter-spacing: 0.5px;
    }
    .section-title .badge {
        font-size: 11px;
        background: #F0F0F0;
        color: #666666;
        padding: 2px 10px;
        border-radius: 10px;
        border: 1px solid #E0E0E0;
    }

    /* ===== 诊断报告卡片 ===== */
    .diagnosis-card {
        background: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-left: 4px solid #111111;
        border-radius: 4px;
        padding: 24px 28px;
        white-space: pre-wrap;
        line-height: 1.9;
        font-size: 14px;
        color: #333333;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
    }
    .diagnosis-card .header {
        font-size: 16px;
        font-weight: 700;
        color: #111111;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid #E0E0E0;
        display: flex;
        align-items: center;
        gap: 8px;
        letter-spacing: 0.5px;
    }

    /* ===== Streamlit 组件覆盖样式 ===== */
    .stButton > button {
        background: #111111;
        color: #FFFFFF;
        font-weight: 500;
        letter-spacing: 1px;
        border: 1px solid #111111;
        border-radius: 4px;
        padding: 11px 24px;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: #333333;
        border-color: #333333;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.20);
    }

    /* 侧边栏(浅灰底,紧凑无滚动条) */
    section[data-testid="stSidebar"] {
        background: #FAFAFA;
        border-right: 1px solid #E0E0E0;
        overflow: hidden !important;
    }
    section[data-testid="stSidebar"] > div {
        overflow: hidden !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0.5rem !important;
    }
    section[data-testid="stSidebar"] .stDateInput,
    section[data-testid="stSidebar"] .stButton,
    section[data-testid="stSidebar"] .stMarkdown {
        margin-bottom: 0.3rem !important;
    }
    section[data-testid="stSidebar"] .stDateInput label,
    section[data-testid="stSidebar"] .stButton label {
        font-size: 0.85rem !important;
    }
    section[data-testid="stSidebar"] .stDateInput input,
    section[data-testid="stSidebar"] .stButton button {
        font-size: 0.85rem !important;
        padding: 0.3rem 0.5rem !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #111111;
        margin-bottom: 0.3rem !important;
        font-size: 1.1rem !important;
    }
    section[data-testid="stSidebar"]::-webkit-scrollbar {
        display: none !important;
        width: 0 !important;
    }
    section[data-testid="stSidebar"] {
        -ms-overflow-style: none !important;
        scrollbar-width: none !important;
    }

    /* 标签与文字颜色 */
    .stMarkdown, .stText { color: #222222; }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #111111;
    }

    /* 输入控件文字 */
    .stDateInput, .stSlider, .stNumberInput, .stTextInput { color: #222222; }
    .stDateInput label, .stSlider label, .stNumberInput label { color: #222222; }

    /* Plotly 图表背景透明 */
    .stPlotlyChart { background: transparent; }

    /* 隐藏默认 Streamlit 元素 */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* 空状态提示 */
    .empty-state {
        text-align: center;
        padding: 80px 20px;
        color: #999999;
    }
    .empty-state .icon {
        margin-bottom: 20px;
        opacity: 0.45;
        display: flex;
        justify-content: center;
    }
    .empty-state .icon svg {
        width: 56px; height: 56px;
        stroke: #666666;
    }
    .empty-state .title {
        font-size: 20px;
        color: #111111;
        margin-bottom: 8px;
        letter-spacing: 0.5px;
    }
    .empty-state .desc {
        font-size: 14px;
        color: #999999;
    }

    /* ===== Lucide 图标样式 ===== */
    .section-title .icon svg,
    .main-header .title-icon svg,
    .diagnosis-card .header svg,
    .sidebar-section svg {
        width: 20px; height: 20px;
        stroke: currentColor;
        stroke-width: 2;
        vertical-align: middle;
    }
    .section-title .icon { display: flex; align-items: center; color: #666666; }
    .main-header .title-icon { display: inline-flex; align-items: center; color: #111111; margin-right: 10px; }
    .main-header .title-icon svg { width: 32px; height: 32px; }
    .diagnosis-card .header svg { color: #666666; margin-right: 4px; }
    .sidebar-section { display: flex; align-items: center; gap: 8px; color: #111111; }
    .metric-card .card-icon svg { width: 16px; height: 16px; }
    </style>
    """, unsafe_allow_html=True)


# ===== Lucide 图标 SVG 路径(内联,不依赖 JS)=====
LUCIDE_ICONS = {
    'sun': '<circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/>',
    'settings': '<path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/>',
    'calendar': '<path d="M8 2v4"/><path d="M16 2v4"/><rect width="18" height="18" x="3" y="4" rx="2"/><path d="M3 10h18"/>',
    'target': '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    'map-pin': '<path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/>',
    'clock': '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    'zap': '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    'bar-chart-3': '<path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/>',
    'database': '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/>',
    'calendar-days': '<path d="M8 2v4"/><path d="M16 2v4"/><rect width="18" height="18" x="3" y="4" rx="2"/><path d="M3 10h18"/><path d="M8 14h.01"/><path d="M12 14h.01"/><path d="M16 14h.01"/><path d="M8 18h.01"/><path d="M12 18h.01"/><path d="M16 18h.01"/>',
    'timer': '<line x1="10" x2="14" y1="2" y2="2"/><line x1="12" x2="15" y1="14" y2="11"/><circle cx="12" cy="14" r="8"/>',
    'trending-up': '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
    'alert-triangle': '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" x2="12" y1="9" y2="13"/><line x1="12" x2="12.01" y1="17" y2="17"/>',
    'percent': '<line x1="19" x2="5" y1="5" y2="19"/><circle cx="6.5" cy="6.5" r="2.5"/><circle cx="17.5" cy="17.5" r="2.5"/>',
    'activity': '<path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2"/>',
    'sliders-horizontal': '<line x1="21" x2="14" y1="4" y2="4"/><line x1="10" x2="3" y1="4" y2="4"/><line x1="21" x2="12" y1="12" y2="12"/><line x1="8" x2="3" y1="12" y2="12"/><line x1="21" x2="16" y1="20" y2="20"/><line x1="12" x2="3" y1="20" y2="20"/><line x1="14" x2="14" y1="2" y2="6"/><line x1="8" x2="8" y1="10" y2="14"/><line x1="16" x2="16" y1="18" y2="22"/>',
    'scale': '<path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/>',
    'alert-octagon': '<path d="M7.86 2h8.28L22 7.86v8.28L16.14 22H7.86L2 16.14V7.86Z"/><path d="M12 8v4"/><path d="M12 16h.01"/>',
    'cloud-sun': '<path d="M12 2v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="M20 12h2"/><path d="m19.07 4.93-1.41 1.41"/><path d="M15.947 12.65a4 4 0 0 0-5.925-4.128"/><path d="M13 22H7a5 5 0 1 1 4.9-6H13a3 3 0 0 1 0 6Z"/>',
    'thermometer': '<path d="M14 4v10.54a4 4 0 1 1-4 0V4a2 2 0 0 1 4 0Z"/>',
    'cloud': '<path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z"/>',
    'sun-medium': '<circle cx="12" cy="12" r="4"/><path d="M12 3v1"/><path d="M12 20v1"/><path d="M3 12h1"/><path d="M20 12h1"/><path d="m18.364 5.636-.707.707"/><path d="m6.343 17.657-.707.707"/><path d="m18.364 18.364-.707-.707"/><path d="m6.343 6.343-.707-.707"/>',
    'droplets': '<path d="M7 16.3c2.2 0 4-1.83 4-4.05 0-1.16-.57-2.26-1.71-3.19S7.29 6.75 7 5.3c-.29 1.45-1.14 2.84-2.29 3.76S3 11.1 3 12.25c0 2.22 1.8 4.05 4 4.05z"/><path d="M12.56 6.6A10.97 10.97 0 0 0 14 3.02c.5 2.5 2 4.9 4 6.5s3 3.5 3 5.5a6.98 6.98 0 0 1-11.91 4.97"/>',
    'bot': '<path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/>',
    'eye': '<path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/>',
}


def lucide_icon(name, size=20, stroke_width=2, color='currentColor'):
    """生成 Lucide 图标的内联 SVG。"""
    paths = LUCIDE_ICONS.get(name, '')
    if not paths:
        return ''
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" '
        f'stroke-linecap="round" stroke-linejoin="round">{paths}</svg>'
    )


def inject_lucide():
    """Lucide 图标已改为内联 SVG,无需注入 JS。此函数保留为空占位。"""
    pass


def render_header():
    """渲染顶部标题栏。"""
    sun_icon = lucide_icon("sun", size=32, color="#111111")
    st.markdown(f"""
    <div class="main-header">
        <h1><span class="title-icon">{sun_icon}</span>光伏电站智能预测与异常诊断平台</h1>
        <p>基于 PatchTST 深度学习模型的光伏功率预测 · Isolation Forest 异常检测 · InternLM 大模型智能诊断</p>
    </div>
    """, unsafe_allow_html=True)


def render_metric_card(label, value, unit='', card_type='primary', icon=None):
    """渲染统计卡片。"""
    icon_html = lucide_icon(icon, size=16) if icon else ''
    st.markdown(f"""
    <div class="metric-card {card_type}">
        <div class="label">{icon_html} {label}</div>
        <div class="value">{value}<span class="unit">{unit}</span></div>
    </div>
    """, unsafe_allow_html=True)


def render_section_title(icon, text, badge=''):
    """渲染分区标题。icon 参数为 Lucide 图标名。"""
    badge_html = f'<span class="badge">{badge}</span>' if badge else ''
    st.markdown(f"""
    <div class="section-title">
        <span class="icon">{lucide_icon(icon)}</span>
        <span class="text">{text}</span>
        {badge_html}
    </div>
    """, unsafe_allow_html=True)


def render_empty_state(icon, title, desc):
    """渲染空状态。icon 参数为 Lucide 图标名。"""
    st.markdown(f"""
    <div class="empty-state">
        <div class="icon">{lucide_icon(icon, size=56, color="#999999")}</div>
        <div class="title">{title}</div>
        <div class="desc">{desc}</div>
    </div>
    """, unsafe_allow_html=True)


def render_diagnosis_card(content):
    """渲染 LLM 诊断报告卡片。"""
    st.markdown(f"""
    <div class="diagnosis-card">
        <div class="header">{lucide_icon("bot", color="#666666")} InternLM 智能诊断报告</div>
        {content}
    </div>
    """, unsafe_allow_html=True)


def render_dispatch_card(dispatch_info):
    """渲染运维决策卡片(黑白极简风格)。

    卡片包含动作名称、动作系数、紧急程度(带颜色标签)与决策依据。
    样式:白底 + 左侧 4px 黑色竖线 + 浅灰细边框,圆角 4px。
    """
    if not dispatch_info:
        return

    action_name = dispatch_info.get('action_name', '')
    coefficient = dispatch_info.get('coefficient', '')
    urgency = dispatch_info.get('urgency', '')
    rationale = dispatch_info.get('rationale', '')

    # 紧急程度颜色映射:低=灰、中=深灰、高=橙、紧急=红
    urgency_color_map = {
        '低': '#666666',
        '中': '#444444',
        '高': '#CC6600',
        '紧急': '#CC0000',
    }
    urgency_color = urgency_color_map.get(urgency, '#666666')

    # 系数展示(转为字符串)
    coef_str = f'{coefficient}' if coefficient != '' else ''

    st.markdown(f"""
    <div style="
        background:#FFFFFF;
        border:1px solid #E0E0E0;
        border-left:4px solid #111111;
        border-radius:4px;
        padding:20px 24px;
        box-shadow:0 1px 3px rgba(0,0,0,0.06);
        margin-bottom:16px;
    ">
        <div style="
            display:flex;
            align-items:center;
            gap:8px;
            margin-bottom:14px;
            padding-bottom:10px;
            border-bottom:1px solid #E0E0E0;
        ">
            <span style="display:inline-flex;align-items:center;color:#666666;">
                {lucide_icon('alert-octagon', size=20, color='#666666')}
            </span>
            <span style="
                font-size:16px;
                font-weight:700;
                color:#111111;
                letter-spacing:0.5px;
            ">运维决策建议</span>
        </div>
        <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:14px;">
            <div style="flex:0 0 auto;">
                <div style="font-size:11px;color:#999999;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:6px;">动作名称</div>
                <div style="
                    font-size:26px;
                    font-weight:700;
                    color:#111111;
                    line-height:1.2;
                ">{action_name}</div>
            </div>
            <div style="flex:0 0 auto;">
                <div style="font-size:11px;color:#999999;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:6px;">动作系数</div>
                <div style="
                    font-size:22px;
                    font-weight:700;
                    color:#666666;
                ">{coef_str}</div>
            </div>
            <div style="flex:0 0 auto;">
                <div style="font-size:11px;color:#999999;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:6px;">紧急程度</div>
                <span style="
                    display:inline-block;
                    padding:4px 14px;
                    border-radius:12px;
                    font-size:13px;
                    font-weight:600;
                    color:#FFFFFF;
                    background:{urgency_color};
                    border:1px solid {urgency_color};
                ">{urgency}</span>
            </div>
        </div>
        <div>
            <div style="font-size:11px;color:#999999;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:6px;">决策依据</div>
            <div style="
                font-size:14px;
                color:#333333;
                line-height:1.8;
            ">{rationale}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


@st.cache_data
def load_data():
    """加载并预处理光伏数据集(带缓存)。"""
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def build_plotly_layout(title='', y_title='功率 (kW)'):
    """构建水墨深色主题的 Plotly 布局配置。"""
    return dict(
        title=dict(text=title, font=dict(color=THEME['text'], size=14), x=0.02),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=THEME['text'], family='system-ui'),
        xaxis=dict(
            title='时间',
            gridcolor=THEME['grid'],
            zerolinecolor=THEME['grid'],
            color=THEME['text_muted'],
            linecolor=THEME['card_border'],
        ),
        yaxis=dict(
            title=y_title,
            gridcolor=THEME['grid'],
            zerolinecolor=THEME['grid'],
            color=THEME['text_muted'],
            linecolor=THEME['card_border'],
        ),
        hovermode='x unified',
        hoverlabel=dict(bgcolor=THEME['card_bg'], font_color=THEME['text'], bordercolor=THEME['card_border']),
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
    inject_lucide()
    render_header()

    # ===== 数据加载 =====
    df = load_data()
    n = len(df)

    # ===== 侧边栏控制 =====
    with st.sidebar:
        st.markdown('<div class="sidebar-section"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg><h3 style="margin:0;color:#111111;">控制面板</h3></div>', unsafe_allow_html=True)
        st.markdown('---')

        # 预测起点:日期选择器
        min_date = df['timestamp'].iloc[0].date()
        max_date = df['timestamp'].iloc[n - SEQ_LEN - PRED_LEN].date()
        default_date = min_date + timedelta(days=180)
        if default_date > max_date:
            default_date = min_date

        st.markdown('<div class="sidebar-section"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 2v4"/><path d="M16 2v4"/><rect width="18" height="18" x="3" y="4" rx="2"/><path d="M3 10h18"/></svg><span style="font-weight:600;color:#111111;">预测起点</span></div>', unsafe_allow_html=True)
        selected_date = st.date_input(
            '历史窗口起始日',
            min_value=min_date,
            max_value=max_date,
            value=default_date,
            help='选择一个日期作为 96 小时历史窗口的起始点,其后须保留至少 120 小时数据。',
            label_visibility='collapsed',
        )

        st.markdown('<div class="sidebar-section"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg><span style="font-weight:600;color:#111111;">异常检测敏感度</span></div>', unsafe_allow_html=True)
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
        st.markdown('<div class="sidebar-section"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/></svg><span style="font-weight:600;color:#111111;">窗口信息</span></div>', unsafe_allow_html=True)
        st.markdown(
            f"<div style='background:#FFFFFF; padding:12px 16px; border-radius:4px; "
            f"border:1px solid #E0E0E0; border-left:3px solid #666666; margin-bottom:8px; font-size:13px;'>"
            f"<div style='color:#999999; font-size:11px; margin-bottom:4px; display:flex;align-items:center;gap:4px;'>"
            f"<svg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><circle cx='12' cy='12' r='10'/><polyline points='12 6 12 12 16 14'/></svg>历史窗口</div>"
            f"<div style='color:#666666;'>{hist_start.strftime('%Y-%m-%d %H:%M')}<br>"
            f"~ {hist_end.strftime('%Y-%m-%d %H:%M')}</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='background:#FFFFFF; padding:12px 16px; border-radius:4px; "
            f"border:1px solid #E0E0E0; border-left:3px solid #111111; margin-bottom:8px; font-size:13px;'>"
            f"<div style='color:#999999; font-size:11px; margin-bottom:4px; display:flex;align-items:center;gap:4px;'>"
            f"<svg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><polygon points='13 2 3 14 12 14 11 22 21 10 12 10 13 2'/></svg>预测时段</div>"
            f"<div style='color:#111111;'>{pred_start.strftime('%Y-%m-%d %H:%M')}<br>"
            f"~ {pred_end.strftime('%Y-%m-%d %H:%M')}</div></div>",
            unsafe_allow_html=True,
        )

        st.markdown('---')
        run_btn = st.button('开始预测与分析', type='primary')

    # ===== 未点击时:展示数据概览 =====
    if not run_btn:
        render_section_title('bar-chart-3', '数据集概览', '就绪')

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_metric_card('数据总量', f'{n:,}', '条', 'info', 'database')
        with col2:
            render_metric_card('起始时间', str(df['timestamp'].iloc[0])[:10], '', 'success', 'calendar-days')
        with col3:
            render_metric_card('结束时间', str(df['timestamp'].iloc[-1])[:10], '', 'success', 'calendar-days')
        with col4:
            render_metric_card('采样间隔', '1', '小时', 'warning', 'timer')

        render_empty_state(
            'sun',
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
    with st.spinner('PatchTST 模型预测中...'):
        try:
            predicted = predict_future(history)
        except Exception as e:
            st.error(f'预测失败: {e}')
            st.error('请确认已运行 train.py 生成模型文件:models/scaler.pkl 和 models/patchtst.pth')
            return

    # 3. 异常检测
    with st.spinner('Isolation Forest 异常检测中...'):
        anomaly_result = detect_anomalies(actual, predicted, contamination=contamination)

    # 3.1 运维决策推荐(基于异常检测结果)
    dispatch_info = recommend_action(anomaly_result)

    # 4. 天气数据
    weather_df = future[['temperature_2m', 'cloud_cover', 'shortwave_radiation', 'relative_humidity_2m']]

    # 5. LLM 诊断(传入 dispatch_info,使报告包含运维动作建议)
    with st.spinner('InternLM 智能诊断生成中...'):
        diagnosis = generate_diagnosis(anomaly_result, weather_df, dispatch_info=dispatch_info)

    # ===== 关键指标概览 =====
    anomaly_indices = anomaly_result['anomaly_indices']
    n_anom = anomaly_result['n_anomalies']
    anom_ratio = anomaly_result['anomaly_ratio']
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    mae = float(np.mean(np.abs(actual - predicted)))

    render_section_title('trending-up', '关键指标', '实时')
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_metric_card('异常点数量', f'{n_anom}', '个', 'danger' if n_anom > 0 else 'success', 'alert-triangle')
    with col2:
        render_metric_card('异常占比', f'{anom_ratio * 100:.2f}', '%', 'danger' if anom_ratio > 0.1 else 'warning', 'percent')
    with col3:
        render_metric_card('RMSE', f'{rmse:.4f}', '', 'info', 'activity')
    with col4:
        render_metric_card('MAE', f'{mae:.4f}', '', 'info', 'activity')
    with col5:
        render_metric_card('检测敏感度', f'{contamination:.2f}', '', 'warning', 'sliders-horizontal')

    # ===== 运维决策建议卡片 =====
    render_section_title('alert-octagon', '运维决策建议', '智能推荐')
    render_dispatch_card(dispatch_info)

    # ===== 模块 1: 功率预测曲线 =====
    render_section_title('zap', '功率预测曲线', '历史 96h + 预测 24h')
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
        fillcolor='rgba(6, 82, 121, 0.07)', layer='below', line_width=0,
    )
    fig1.update_layout(build_plotly_layout('功率预测曲线'), height=420)
    st.plotly_chart(fig1, use_container_width=True)

    # ===== 模块 2 & 3: 并排展示对比图 + 异常标记图 =====
    col_left, col_right = st.columns(2)

    with col_left:
        render_section_title('scale', '实际 vs 预测', '对比')
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=future_ts, y=actual, mode='lines+markers', name='实际功率',
            line=dict(color=THEME['success'], width=2.5),
            marker=dict(size=6),
            fill='tozeroy', fillcolor='rgba(80, 97, 109, 0.10)',
        ))
        fig2.add_trace(go.Scatter(
            x=future_ts, y=predicted, mode='lines+markers', name='预测功率',
            line=dict(color=THEME['warning'], width=2.5, dash='dash'),
            marker=dict(size=6),
        ))
        fig2.update_layout(build_plotly_layout('实际 vs 预测对比'), height=400)
        st.plotly_chart(fig2, use_container_width=True)

    with col_right:
        render_section_title('alert-octagon', '异常点标记', f'{n_anom} 个异常')
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
        fig3.update_layout(build_plotly_layout('异常点检测'), height=400)
        st.plotly_chart(fig3, use_container_width=True)

    # ===== 模块 4: 天气环境数据 =====
    render_section_title('cloud-sun', '天气环境数据', '预测时段')
    col_w1, col_w2, col_w3, col_w4 = st.columns(4)
    with col_w1:
        render_metric_card('温度', f'{weather_df["temperature_2m"].mean():.1f}', '°C', 'info', 'thermometer')
    with col_w2:
        render_metric_card('云量', f'{weather_df["cloud_cover"].mean():.1f}', '%', 'warning', 'cloud')
    with col_w3:
        render_metric_card('短波辐射', f'{weather_df["shortwave_radiation"].mean():.1f}', 'W/m²', 'primary', 'sun-medium')
    with col_w4:
        render_metric_card('相对湿度', f'{weather_df["relative_humidity_2m"].mean():.1f}', '%', 'info', 'droplets')

    # ===== 模块 5: LLM 诊断报告 =====
    render_section_title('bot', 'LLM 智能诊断', 'InternLM')
    # 将诊断文本格式化为 HTML(保留换行,转义特殊字符)
    diagnosis_html = diagnosis.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    diagnosis_html = diagnosis_html.replace('\n', '<br>')
    render_diagnosis_card(diagnosis_html)


if __name__ == '__main__':
    main()
