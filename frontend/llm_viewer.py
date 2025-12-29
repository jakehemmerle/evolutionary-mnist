"""LLM decisions viewer tab."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st


def load_llm_decisions(experiment_dir: Path) -> dict | None:
    """Load llm_decisions.json from an experiment directory."""
    decisions_file = experiment_dir / "llm_decisions.json"
    if decisions_file.exists():
        with open(decisions_file) as f:
            return json.load(f)
    return None


def load_config(experiment_dir: Path) -> dict | None:
    """Load config.json from an experiment directory."""
    config_file = experiment_dir / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            return json.load(f)
    return None


def render_llm_viewer_tab(experiments_dir: Path) -> None:
    """Render the LLM decisions viewer tab."""
    st.header("LLM Advisor Decisions")
    
    # Find experiments with LLM decisions
    experiments_with_llm = []
    for exp_dir in sorted(experiments_dir.iterdir(), reverse=True):
        if exp_dir.is_dir():
            decisions = load_llm_decisions(exp_dir)
            config = load_config(exp_dir)
            if decisions:
                experiments_with_llm.append({
                    "name": exp_dir.name,
                    "path": exp_dir,
                    "decisions": decisions,
                    "config": config,
                })
    
    if not experiments_with_llm:
        st.info(
            "No LLM decision logs found. Run an experiment with LLM enabled "
            "(`evolution.llm.enabled = true` in config) to see decisions."
        )
        return
    
    # Experiment selector
    selected_name = st.selectbox(
        "Select Experiment",
        [exp["name"] for exp in experiments_with_llm],
        key="llm_experiment_selector",
    )
    
    selected_exp = next(exp for exp in experiments_with_llm if exp["name"] == selected_name)
    decisions = selected_exp["decisions"].get("decisions", [])
    config = selected_exp.get("config", {})
    
    # Show LLM configuration
    llm_config = config.get("evolution", {}).get("llm", {})
    if llm_config:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model", llm_config.get("model", "Unknown"))
        with col2:
            st.metric("Temperature", llm_config.get("temperature", 0.2))
        with col3:
            st.metric("Total Decisions", len(decisions))
    
    st.divider()
    
    if not decisions:
        st.info("No decisions recorded yet.")
        return
    
    # Timeline view
    st.subheader("Decision Timeline")
    
    for decision in sorted(decisions, key=lambda d: d.get("generation", 0)):
        _render_decision_card(decision)


def _render_decision_card(decision: dict) -> None:
    """Render a single LLM decision as an expandable card."""
    generation = decision.get("generation", 0)
    fallback_used = decision.get("fallback_used", False)
    attempts = decision.get("attempts", 1)
    timestamp = decision.get("timestamp", "")
    
    # Status indicator
    if fallback_used:
        status_icon = "⚠️"
        status_text = "Fallback Used"
    else:
        status_icon = "✅"
        status_text = "LLM Success"
    
    # Card header
    header = f"Generation {generation} - {status_icon} {status_text}"
    if attempts > 1:
        header += f" (after {attempts} attempts)"
    
    with st.expander(header, expanded=generation == 1):
        # Timestamp and status
        col1, col2 = st.columns([3, 1])
        with col1:
            if timestamp:
                st.caption(f"🕐 {timestamp}")
        with col2:
            if fallback_used:
                st.error("Fallback", icon="⚠️")
            else:
                st.success("LLM", icon="✅")
        
        # Option lists that were chosen
        option_lists = decision.get("validated_option_lists")
        if option_lists:
            st.subheader("Parameters Explored")
            
            # Highlight varied parameters
            varied = []
            fixed = []
            for param, values in option_lists.items():
                if len(values) > 1:
                    varied.append({"Parameter": param, "Values": str(values), "Status": "🔀 Varied"})
                else:
                    fixed.append({"Parameter": param, "Values": str(values), "Status": "📌 Fixed"})
            
            if varied:
                st.markdown("**Varied Parameters:**")
                st.table(varied)
            
            if fixed:
                with st.expander("Fixed Parameters"):
                    st.table(fixed)
        
        # Error if fallback was used
        error = decision.get("error")
        if error:
            st.error(f"**Error:** {error}")
        
        # Prompts and response (collapsible)
        with st.expander("View Prompts & Response"):
            st.markdown("**System Prompt:**")
            st.code(decision.get("system_prompt", "N/A"), language=None)
            
            st.markdown("**User Prompt:**")
            user_prompt = decision.get("user_prompt", "N/A")
            try:
                # Try to pretty-print JSON
                parsed = json.loads(user_prompt)
                st.json(parsed)
            except (json.JSONDecodeError, TypeError):
                st.code(user_prompt, language=None)
            
            st.markdown("**LLM Response:**")
            raw_response = decision.get("raw_response")
            if raw_response:
                try:
                    parsed = json.loads(raw_response)
                    st.json(parsed)
                except (json.JSONDecodeError, TypeError):
                    st.code(raw_response, language=None)
            else:
                st.caption("No response received")
        
        st.divider()

