"""Interactive visualizations using Plotly."""

from __future__ import annotations

import json
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def load_results(experiment_dir: Path) -> dict | None:
    """Load results.json from an experiment directory."""
    results_file = experiment_dir / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    return None


def render_visualizations_tab(experiments_dir: Path) -> None:
    """Render the interactive visualizations tab."""
    st.header("Results Visualization")
    
    # Get all experiment directories with results
    experiments = []
    for exp_dir in sorted(experiments_dir.iterdir(), reverse=True):
        if exp_dir.is_dir():
            results = load_results(exp_dir)
            if results:
                experiments.append({"name": exp_dir.name, "results": results})
    
    if not experiments:
        st.info("No experiments with results found.")
        return
    
    # Experiment selector
    selected_name = st.selectbox(
        "Select Experiment",
        [exp["name"] for exp in experiments],
        key="viz_experiment_selector",
    )
    
    selected_exp = next(exp for exp in experiments if exp["name"] == selected_name)
    results = selected_exp["results"].get("results", [])
    
    if not results:
        st.warning("No results found in this experiment.")
        return
    
    # Visualization tabs
    viz_tab1, viz_tab2, viz_tab3 = st.tabs([
        "Pareto Front",
        "Loss Curves", 
        "Parameter Analysis",
    ])
    
    with viz_tab1:
        _render_pareto_front(results)
    
    with viz_tab2:
        _render_loss_curves(results)
    
    with viz_tab3:
        _render_parameter_analysis(results)


def _render_pareto_front(results: list[dict]) -> None:
    """Render interactive Pareto front scatter plot."""
    st.subheader("Pareto Front: Accuracy vs Training Time")
    
    # Prepare data
    data = []
    for i, r in enumerate(results):
        params = r.get("params", {})
        data.append({
            "Accuracy": r.get("val_accuracy", 0) * 100,
            "Wall Time (s)": r.get("wall_time_seconds", 0),
            "Model": params.get("model_type", "mlp"),
            "LR": params.get("learning_rate", 0),
            "Epochs": params.get("epochs", 0),
            "Param Count": r.get("param_count", 0),
            "Config": i + 1,
        })
    
    # Create scatter plot
    fig = px.scatter(
        data,
        x="Wall Time (s)",
        y="Accuracy",
        color="Model",
        size="Param Count",
        hover_data=["LR", "Epochs", "Config"],
        title="Accuracy vs Training Time",
    )
    
    # Highlight Pareto front
    pareto_points = _compute_pareto_front(data)
    if pareto_points:
        pareto_x = [p["Wall Time (s)"] for p in pareto_points]
        pareto_y = [p["Accuracy"] for p in pareto_points]
        
        # Sort by x for line
        sorted_pareto = sorted(zip(pareto_x, pareto_y))
        pareto_x, pareto_y = zip(*sorted_pareto)
        
        fig.add_trace(go.Scatter(
            x=pareto_x,
            y=pareto_y,
            mode="lines",
            name="Pareto Front",
            line=dict(color="rgba(255, 0, 0, 0.5)", dash="dash", width=2),
        ))
    
    fig.update_layout(
        height=500,
        xaxis_title="Training Time (seconds)",
        yaxis_title="Validation Accuracy (%)",
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _compute_pareto_front(data: list[dict]) -> list[dict]:
    """Compute Pareto front (maximize accuracy, minimize time)."""
    pareto = []
    for point in data:
        dominated = False
        for other in data:
            # Other dominates if: higher accuracy AND lower time
            if (other["Accuracy"] > point["Accuracy"] and 
                other["Wall Time (s)"] <= point["Wall Time (s)"]):
                dominated = True
                break
            if (other["Accuracy"] >= point["Accuracy"] and 
                other["Wall Time (s)"] < point["Wall Time (s)"]):
                dominated = True
                break
        if not dominated:
            pareto.append(point)
    return pareto


def _render_loss_curves(results: list[dict]) -> None:
    """Render interactive loss curves with selection."""
    st.subheader("Training Loss Curves")
    
    # Selection options
    col1, col2 = st.columns([2, 1])
    with col1:
        show_mode = st.radio(
            "Show",
            ["Top 10 by Accuracy", "All Configs", "Select Specific"],
            horizontal=True,
        )
    
    # Filter results based on selection
    if show_mode == "Top 10 by Accuracy":
        sorted_results = sorted(results, key=lambda x: x.get("val_accuracy", 0), reverse=True)
        display_results = sorted_results[:10]
    elif show_mode == "All Configs":
        display_results = results
    else:
        # Allow specific selection
        config_options = [f"Config {i+1}" for i in range(len(results))]
        selected_configs = st.multiselect(
            "Select configurations",
            config_options,
            default=config_options[:5] if len(config_options) >= 5 else config_options,
        )
        indices = [int(c.split()[-1]) - 1 for c in selected_configs]
        display_results = [results[i] for i in indices if i < len(results)]
    
    if not display_results:
        st.info("No configurations selected.")
        return
    
    # Create loss curves plot
    fig = go.Figure()
    
    for i, r in enumerate(display_results):
        epoch_losses = r.get("epoch_losses", [])
        params = r.get("params", {})
        accuracy = r.get("val_accuracy", 0) * 100
        
        model_type = params.get("model_type", "mlp")
        lr = params.get("learning_rate", 0)
        
        label = f"{model_type} lr={lr} ({accuracy:.1f}%)"
        
        fig.add_trace(go.Scatter(
            x=list(range(1, len(epoch_losses) + 1)),
            y=epoch_losses,
            mode="lines+markers",
            name=label,
        ))
    
    fig.update_layout(
        height=500,
        xaxis_title="Epoch",
        yaxis_title="Training Loss",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
        ),
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_parameter_analysis(results: list[dict]) -> None:
    """Render parameter importance analysis."""
    st.subheader("Parameter Analysis")
    
    # Extract all parameters
    param_names = set()
    for r in results:
        param_names.update(r.get("params", {}).keys())
    
    # Let user select parameter to analyze
    selected_param = st.selectbox(
        "Analyze Parameter",
        sorted(param_names),
        key="param_analysis_selector",
    )
    
    # Collect data for selected parameter
    param_data = []
    for r in results:
        params = r.get("params", {})
        value = params.get(selected_param)
        if value is not None:
            param_data.append({
                "Value": str(value),
                "Accuracy": r.get("val_accuracy", 0) * 100,
                "Wall Time": r.get("wall_time_seconds", 0),
            })
    
    if not param_data:
        st.info(f"No data for parameter '{selected_param}'")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Box plot of accuracy by parameter value
        fig1 = px.box(
            param_data,
            x="Value",
            y="Accuracy",
            title=f"Accuracy by {selected_param}",
        )
        fig1.update_layout(height=350)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Box plot of training time by parameter value
        fig2 = px.box(
            param_data,
            x="Value",
            y="Wall Time",
            title=f"Training Time by {selected_param}",
        )
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Parameter heatmap (if we have enough varied parameters)
    _render_parameter_heatmap(results, param_names)


def _render_parameter_heatmap(results: list[dict], param_names: set) -> None:
    """Render a heatmap showing parameter correlations with accuracy."""
    st.subheader("Parameter Correlation Heatmap")
    
    # Find numeric parameters
    numeric_params = []
    for param in param_names:
        values = [r.get("params", {}).get(param) for r in results]
        if all(isinstance(v, (int, float)) for v in values if v is not None):
            numeric_params.append(param)
    
    if len(numeric_params) < 2:
        st.info("Need at least 2 numeric parameters for heatmap.")
        return
    
    # Select two parameters for heatmap
    col1, col2 = st.columns(2)
    with col1:
        param_x = st.selectbox("X-axis Parameter", numeric_params, key="heatmap_x")
    with col2:
        remaining = [p for p in numeric_params if p != param_x]
        param_y = st.selectbox("Y-axis Parameter", remaining, key="heatmap_y")
    
    # Create heatmap data
    heatmap_data = []
    for r in results:
        params = r.get("params", {})
        x_val = params.get(param_x)
        y_val = params.get(param_y)
        if x_val is not None and y_val is not None:
            heatmap_data.append({
                param_x: x_val,
                param_y: y_val,
                "Accuracy": r.get("val_accuracy", 0) * 100,
            })
    
    if heatmap_data:
        fig = px.scatter(
            heatmap_data,
            x=param_x,
            y=param_y,
            color="Accuracy",
            size="Accuracy",
            color_continuous_scale="viridis",
            title=f"Accuracy by {param_x} and {param_y}",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

