"""Frontend module for MNIST evolution monitoring dashboard."""

from frontend.history import render_history_tab
from frontend.live_monitor import render_live_monitor_tab
from frontend.llm_viewer import render_llm_viewer_tab
from frontend.visualizations import render_visualizations_tab

__all__ = [
    "render_live_monitor_tab",
    "render_history_tab",
    "render_visualizations_tab",
    "render_llm_viewer_tab",
]

