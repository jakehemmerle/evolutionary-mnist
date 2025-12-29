"""Experiment history browser tab."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st


def load_results(experiment_dir: Path) -> dict | None:
    """Load results.json from an experiment directory."""
    results_file = experiment_dir / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    return None


def load_config(experiment_dir: Path) -> dict | None:
    """Load config.json from an experiment directory."""
    config_file = experiment_dir / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            return json.load(f)
    return None


def render_history_tab(experiments_dir: Path) -> None:
    """Render the experiment history browser tab."""
    st.header("Experiment History")
    
    # Get all experiment directories
    experiments = []
    for exp_dir in sorted(experiments_dir.iterdir(), reverse=True):
        if exp_dir.is_dir():
            results = load_results(exp_dir)
            config = load_config(exp_dir)
            if results:
                experiments.append({
                    "name": exp_dir.name,
                    "path": exp_dir,
                    "results": results,
                    "config": config,
                })
    
    if not experiments:
        st.info("No completed experiments found. Run an experiment first.")
        return
    
    # Experiment selector
    experiment_names = [exp["name"] for exp in experiments]
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_name = st.selectbox(
            "Select Experiment",
            experiment_names,
            key="history_experiment_selector",
        )
    with col2:
        compare_mode = st.toggle("Compare Mode", value=False)
    
    selected_exp = next(exp for exp in experiments if exp["name"] == selected_name)
    
    if compare_mode:
        _render_comparison_view(experiments)
    else:
        _render_single_experiment(selected_exp)


def _render_single_experiment(experiment: dict) -> None:
    """Render detailed view of a single experiment."""
    results = experiment["results"]
    config = experiment["config"]
    
    # Summary metrics
    st.subheader("Summary")
    
    all_results = results.get("results", [])
    if all_results:
        best_result = max(all_results, key=lambda x: x.get("val_accuracy", 0))
        avg_accuracy = sum(r.get("val_accuracy", 0) for r in all_results) / len(all_results)
        total_time = sum(r.get("wall_time_seconds", 0) for r in all_results)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Configs", len(all_results))
        with col2:
            st.metric("Best Accuracy", f"{best_result['val_accuracy'] * 100:.2f}%")
        with col3:
            st.metric("Avg Accuracy", f"{avg_accuracy * 100:.2f}%")
        with col4:
            st.metric("Total Time", f"{total_time:.1f}s")
    
    # Configuration details
    if config:
        with st.expander("Experiment Configuration", expanded=False):
            st.json(config)
    
    # Results table
    st.subheader("All Configurations")
    
    # Convert to display format
    table_data = []
    for i, r in enumerate(sorted(all_results, key=lambda x: x.get("val_accuracy", 0), reverse=True)):
        params = r.get("params", {})
        row = {
            "Rank": i + 1,
            "Accuracy": f"{r.get('val_accuracy', 0) * 100:.2f}%",
            "Loss": f"{r.get('val_loss', 0):.4f}",
            "Time": f"{r.get('wall_time_seconds', 0):.1f}s",
            "LR": params.get("learning_rate", 0),
            "Epochs": params.get("epochs", 0),
            "Channels": params.get("cnn_channels", "-"),
            "Kernel": params.get("cnn_kernel_size", "-"),
            "FC": params.get("cnn_fc_hidden", "-"),
        }
        table_data.append(row)
    
    st.dataframe(
        table_data,
        use_container_width=True,
        hide_index=True,
    )
    
    # Show charts if available
    charts_dir = experiment["path"] / "charts"
    if charts_dir.exists():
        st.subheader("Charts")
        
        chart_files = list(charts_dir.glob("*.png"))
        if chart_files:
            cols = st.columns(2)
            for i, chart_file in enumerate(chart_files):
                with cols[i % 2]:
                    st.image(str(chart_file), caption=chart_file.stem.replace("_", " ").title())


def _render_comparison_view(experiments: list[dict]) -> None:
    """Render comparison view of multiple experiments."""
    st.subheader("Compare Experiments")
    
    # Multi-select for comparison
    selected_names = st.multiselect(
        "Select experiments to compare",
        [exp["name"] for exp in experiments],
        default=[experiments[0]["name"]] if experiments else [],
    )
    
    if len(selected_names) < 2:
        st.info("Select at least 2 experiments to compare.")
        return
    
    selected_exps = [exp for exp in experiments if exp["name"] in selected_names]
    
    # Comparison table
    comparison_data = []
    for exp in selected_exps:
        results = exp["results"].get("results", [])
        if results:
            best = max(results, key=lambda x: x.get("val_accuracy", 0))
            avg_acc = sum(r.get("val_accuracy", 0) for r in results) / len(results)
            total_time = sum(r.get("wall_time_seconds", 0) for r in results)
            
            comparison_data.append({
                "Experiment": exp["name"],
                "Configs": len(results),
                "Best Acc": f"{best['val_accuracy'] * 100:.2f}%",
                "Avg Acc": f"{avg_acc * 100:.2f}%",
                "Total Time": f"{total_time:.1f}s",
            })
    
    st.dataframe(comparison_data, use_container_width=True, hide_index=True)

