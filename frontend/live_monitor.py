"""Live monitoring tab for real-time training progress."""

from __future__ import annotations

import json
import time
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st


def load_progress(experiment_dir: Path) -> dict | None:
    """Load progress.json from an experiment directory."""
    progress_file = experiment_dir / "progress.json"
    if progress_file.exists():
        with open(progress_file) as f:
            return json.load(f)
    return None


def render_live_monitor_tab(experiments_dir: Path) -> None:
    """Render the live monitoring tab."""
    st.header("Live Training Monitor")
    
    # Find experiments with progress files
    running_experiments = []
    for exp_dir in experiments_dir.iterdir():
        if exp_dir.is_dir():
            progress = load_progress(exp_dir)
            if progress and progress.get("status") == "running":
                running_experiments.append((exp_dir.name, progress))
    
    if not running_experiments:
        # Check for any experiments at all
        all_experiments = [d for d in experiments_dir.iterdir() if d.is_dir()]
        if not all_experiments:
            st.info("No experiments found. Run an experiment to see live progress.")
        else:
            st.info("No experiments currently running. Start an evolution experiment to monitor progress.")
            
            # Show most recently completed experiment
            completed = []
            for exp_dir in all_experiments:
                progress = load_progress(exp_dir)
                if progress:
                    completed.append((exp_dir.name, progress))
            
            if completed:
                st.subheader("Most Recent Experiment")
                name, progress = completed[-1]
                _render_experiment_status(name, progress, is_live=False)
        return
    
    # Auto-refresh toggle
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption("Monitoring active experiments...")
    with col2:
        auto_refresh = st.toggle("Auto-refresh", value=True)
    
    # Render each running experiment
    for name, progress in running_experiments:
        _render_experiment_status(name, progress, is_live=True)
    
    # Auto-refresh mechanism
    if auto_refresh and running_experiments:
        time.sleep(2)
        st.rerun()


def _render_experiment_status(name: str, progress: dict, is_live: bool) -> None:
    """Render status card for a single experiment."""
    status = progress.get("status", "unknown")
    status_emoji = "🔄" if status == "running" else "✅" if status == "completed" else "⚠️"
    
    with st.container():
        st.subheader(f"{status_emoji} {name}")
        
        # Generation progress
        gen = progress.get("generation", {})
        current_gen = gen.get("current", 0)
        total_gen = gen.get("total", 1)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Generation", f"{current_gen} / {total_gen}")
        with col2:
            st.metric("Configs Evaluated", progress.get("total_configs_evaluated", 0))
        with col3:
            best_acc = progress.get("best_accuracy", 0)
            st.metric("Best Accuracy", f"{best_acc * 100:.2f}%")
        
        # Progress bar
        progress_pct = current_gen / total_gen if total_gen > 0 else 0
        st.progress(progress_pct, text=f"Generation {current_gen} of {total_gen}")
        
        # Current configs being explored
        current_configs = progress.get("current_configs")
        if current_configs and is_live:
            with st.expander("Current Generation Configs", expanded=True):
                st.dataframe(current_configs)
        
        # Live loss curves from results so far
        results = progress.get("results_so_far", [])
        if results:
            with st.expander("Training Progress", expanded=is_live):
                _render_live_loss_curves(results)
        
        st.divider()


def _render_live_loss_curves(results: list[dict]) -> None:
    """Render streaming loss curves from results."""
    fig = go.Figure()
    
    # Show last 10 configs for readability
    recent_results = results[-10:]
    
    for i, result in enumerate(recent_results):
        epoch_losses = result.get("epoch_losses", [])
        params = result.get("params", {})

        # Create label from key params
        lr = params.get("learning_rate", 0)
        channels = params.get("cnn_channels", 32)
        label = f"ch={channels} lr={lr}"

        fig.add_trace(go.Scatter(
            x=list(range(1, len(epoch_losses) + 1)),
            y=epoch_losses,
            mode="lines+markers",
            name=label,
            opacity=0.7,
        ))
    
    fig.update_layout(
        title="Training Loss (Recent Configs)",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        height=350,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.4,
            xanchor="center",
            x=0.5,
        ),
    )
    
    st.plotly_chart(fig, use_container_width=True)

