#!/usr/bin/env python3
"""
MNIST Evolution Dashboard

A Streamlit dashboard for monitoring and analyzing hyperparameter evolution experiments.

Run with:
    uv run streamlit run frontend/app.py
"""

from pathlib import Path

import streamlit as st

from history import render_history_tab
from live_monitor import render_live_monitor_tab
from llm_viewer import render_llm_viewer_tab
from visualizations import render_visualizations_tab

# Page configuration
st.set_page_config(
    page_title="MNIST Evolution Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for dark-mode friendly aesthetic
st.markdown("""
<style>
    /* Import distinctive fonts */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Root variables for theming */
    :root {
        --accent-primary: #00d4aa;
        --accent-secondary: #7c3aed;
        --accent-warning: #f59e0b;
        --surface-elevated: rgba(255, 255, 255, 0.03);
        --border-subtle: rgba(255, 255, 255, 0.08);
    }
    
    /* Main container styling */
    .stApp {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Headers with gradient accent */
    h1 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #00d4aa 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        padding-bottom: 0.5rem;
    }
    
    h2, h3 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        color: #e2e8f0;
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background: var(--surface-elevated);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 1rem;
    }
    
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.8rem !important;
        color: var(--accent-primary) !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 400;
        text-transform: uppercase;
        font-size: 0.75rem !important;
        letter-spacing: 0.05em;
    }
    
    /* Tables */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Outfit', sans-serif;
        font-weight: 500;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        background: var(--surface-elevated);
        border: 1px solid var(--border-subtle);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 212, 170, 0.15) 0%, rgba(124, 58, 237, 0.15) 100%) !important;
        border-color: var(--accent-primary) !important;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        font-family: 'Outfit', sans-serif;
        font-weight: 500;
        background: var(--surface-elevated);
        border-radius: 8px;
    }
    
    /* Code blocks */
    code {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem;
    }
    
    /* Buttons */
    .stButton > button {
        font-family: 'Outfit', sans-serif;
        font-weight: 500;
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 212, 170, 0.2);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary));
    }
    
    /* Dividers */
    hr {
        border-color: var(--border-subtle);
        margin: 1.5rem 0;
    }
    
    /* Info/Warning boxes */
    .stAlert {
        border-radius: 8px;
        border-left-width: 4px;
    }
    
    /* Selectbox */
    .stSelectbox [data-baseweb="select"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Experiments directory
EXPERIMENTS_DIR = Path("experiments")


def main():
    """Main dashboard entry point."""
    # Header
    st.title("🧬 MNIST Evolution Dashboard")
    st.caption("Monitor training progress and analyze hyperparameter evolution experiments")
    
    # Main navigation tabs
    tab_live, tab_history, tab_viz, tab_llm = st.tabs([
        "📡 Live Monitor",
        "📚 History",
        "📊 Visualizations",
        "🤖 LLM Decisions",
    ])
    
    with tab_live:
        render_live_monitor_tab(EXPERIMENTS_DIR)
    
    with tab_history:
        render_history_tab(EXPERIMENTS_DIR)
    
    with tab_viz:
        render_visualizations_tab(EXPERIMENTS_DIR)
    
    with tab_llm:
        render_llm_viewer_tab(EXPERIMENTS_DIR)


if __name__ == "__main__":
    main()

