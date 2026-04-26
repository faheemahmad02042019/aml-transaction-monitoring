"""
Streamlit Compliance Dashboard for AML Transaction Monitoring.

Provides an interactive web interface for compliance analysts:

  - Alert queue with priority sorting and severity filtering
  - Individual alert detail view with LLM-generated summary
  - Transaction network graph visualization (Plotly + NetworkX)
  - Alert statistics: by type, severity, disposition over time
  - Model performance monitoring charts
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers (used when Streamlit is running)
# ─────────────────────────────────────────────────────────────────────────────

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
ALERTS_DIR = Path(__file__).resolve().parent.parent / "alerts"


def load_sample_alerts() -> pd.DataFrame:
    """
    Load alerts from the outputs directory, or generate sample data
    for demonstration purposes.

    Returns
    -------
    pd.DataFrame
        Alert data with columns for the dashboard.
    """
    # Try to load from exported alerts
    for pattern in ["alerts/*.json", "outputs/*alert*.json", "outputs/*alert*.parquet"]:
        files = sorted(Path(__file__).resolve().parent.parent.glob(pattern))
        if files:
            latest = files[-1]
            try:
                if latest.suffix == ".json":
                    with open(latest, "r") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        return pd.DataFrame(data)
                elif latest.suffix == ".parquet":
                    return pd.read_parquet(latest)
            except Exception as e:
                logger.warning(f"Failed to load alerts from {latest}: {e}")

    # Generate sample data for demonstration
    return _generate_sample_alerts()


def _generate_sample_alerts() -> pd.DataFrame:
    """Generate sample alert data for dashboard demonstration."""
    rng = np.random.RandomState(42)
    n_alerts = 50

    severities = rng.choice(["HIGH", "MEDIUM", "LOW"], size=n_alerts, p=[0.2, 0.4, 0.4])
    statuses = rng.choice(
        ["OPEN", "IN_REVIEW", "ESCALATED", "CLOSED"],
        size=n_alerts,
        p=[0.4, 0.25, 0.15, 0.2],
    )
    dispositions = []
    for s in statuses:
        if s == "CLOSED":
            dispositions.append(rng.choice(["CLOSED_NO_ACTION", "SAR_FILED"]))
        elif s == "ESCALATED":
            dispositions.append("SAR_FILED")
        else:
            dispositions.append(None)

    rules_pool = [
        "structuring", "rapid_movement", "round_tripping",
        "geographic_risk", "dormant_reactivation", "velocity",
    ]
    triggered_rules = []
    for _ in range(n_alerts):
        n_rules = rng.randint(1, 4)
        triggered_rules.append(list(rng.choice(rules_pool, size=n_rules, replace=False)))

    now = datetime.utcnow()
    created_dates = [
        now - timedelta(hours=rng.randint(1, 168)) for _ in range(n_alerts)
    ]

    llm_dispositions = rng.choice(
        ["ESCALATE_TO_SAR", "CONTINUE_MONITORING", "CLOSE"],
        size=n_alerts,
        p=[0.3, 0.5, 0.2],
    )

    records = []
    for i in range(n_alerts):
        score = rng.beta(2, 5) if severities[i] == "LOW" else (
            rng.beta(5, 2) if severities[i] == "HIGH" else rng.beta(3, 3)
        )
        score = float(np.clip(score, 0.1, 0.99))

        records.append({
            "alert_id": f"ALT-{i + 1:04d}",
            "entity_id": f"BANK_{rng.randint(1, 20):04d}_{rng.randint(10000, 99999)}",
            "severity": severities[i],
            "ensemble_score": round(score, 3),
            "ml_score": round(float(rng.beta(3, 3)), 3),
            "graph_score": round(float(rng.beta(2, 4)), 3),
            "triggered_rules": triggered_rules[i],
            "transaction_count": int(rng.randint(1, 25)),
            "total_amount": round(float(rng.lognormal(10, 1.5)), 2),
            "status": statuses[i],
            "disposition": dispositions[i],
            "created_at": created_dates[i].isoformat(),
            "llm_summary": (
                f"Account {records[-1]['entity_id'] if records else 'N/A'} exhibited suspicious "
                f"activity patterns consistent with {rng.choice(rules_pool)}. "
                f"Multiple transactions totaling significant amounts were observed "
                f"within a short timeframe."
            ) if i < n_alerts else "",
            "llm_recommendation": llm_dispositions[i],
            "confidence_level": rng.choice(["HIGH", "MEDIUM", "LOW"]),
            "typology": rng.choice([
                "STRUCTURING", "LAYERING", "ROUND_TRIPPING",
                "RAPID_MOVEMENT", "MIXED", "UNKNOWN",
            ]),
        })

    return pd.DataFrame(records)


def load_model_metrics() -> Dict[str, Any]:
    """
    Load model performance metrics, or return sample data.

    Returns
    -------
    Dict[str, Any]
        Model performance metrics.
    """
    metrics_files = sorted(OUTPUTS_DIR.glob("evaluation_*.json"))
    if metrics_files:
        try:
            with open(metrics_files[-1], "r") as f:
                return json.load(f)
        except Exception:
            pass

    return {
        "transaction_level_LightGBM": {
            "auc_roc": 0.97,
            "auc_pr": 0.58,
            "f1": 0.61,
            "precision": 0.52,
            "recall": 0.73,
            "recall_at_1pct_fpr": 0.73,
            "recall_at_5pct_fpr": 0.88,
        },
        "system_comparison": {
            "system_metrics": {
                "rules_only": {"auc_roc": 0.82, "auc_pr": 0.15, "f1": 0.22, "recall_at_5pct_fpr": 0.58},
                "ml_only": {"auc_roc": 0.94, "auc_pr": 0.42, "f1": 0.48, "recall_at_5pct_fpr": 0.78},
                "hybrid": {"auc_roc": 0.97, "auc_pr": 0.58, "f1": 0.61, "recall_at_5pct_fpr": 0.88},
            }
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard application
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Main entry point for the Streamlit dashboard."""
    try:
        import streamlit as st
        import plotly.express as px
        import plotly.graph_objects as go
        import networkx as nx
    except ImportError as e:
        print(f"Required packages not installed: {e}")
        print("Install with: pip install streamlit plotly networkx")
        return

    # ─── Page configuration ──────────────────────────────────────────
    st.set_page_config(
        page_title="AML Compliance Dashboard",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("AML Transaction Monitoring Dashboard")
    st.markdown("**Compliance Alert Queue & Performance Monitoring**")

    # ─── Sidebar ─────────────────────────────────────────────────────
    st.sidebar.header("Filters")
    alerts_df = load_sample_alerts()

    severity_filter = st.sidebar.multiselect(
        "Severity",
        options=["HIGH", "MEDIUM", "LOW"],
        default=["HIGH", "MEDIUM", "LOW"],
    )
    status_filter = st.sidebar.multiselect(
        "Status",
        options=["OPEN", "IN_REVIEW", "ESCALATED", "CLOSED"],
        default=["OPEN", "IN_REVIEW", "ESCALATED"],
    )
    min_score = st.sidebar.slider("Minimum Ensemble Score", 0.0, 1.0, 0.0, 0.05)

    # Apply filters
    filtered = alerts_df[
        (alerts_df["severity"].isin(severity_filter))
        & (alerts_df["status"].isin(status_filter))
        & (alerts_df["ensemble_score"] >= min_score)
    ].sort_values("ensemble_score", ascending=False)

    # ─── Tab layout ──────────────────────────────────────────────────
    tab_queue, tab_detail, tab_network, tab_stats, tab_performance = st.tabs([
        "Alert Queue",
        "Alert Detail",
        "Network Graph",
        "Alert Statistics",
        "Model Performance",
    ])

    # ═══════════════════════════════════════════════════════════════════
    # TAB 1: Alert Queue
    # ═══════════════════════════════════════════════════════════════════
    with tab_queue:
        st.header("Prioritized Alert Queue")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Alerts", len(filtered))
        col2.metric(
            "HIGH Severity",
            len(filtered[filtered["severity"] == "HIGH"]),
        )
        col3.metric(
            "Open Alerts",
            len(filtered[filtered["status"] == "OPEN"]),
        )
        col4.metric(
            "Avg Score",
            f"{filtered['ensemble_score'].mean():.3f}" if len(filtered) > 0 else "N/A",
        )

        # Display table
        display_cols = [
            "alert_id", "entity_id", "severity", "ensemble_score",
            "ml_score", "graph_score", "transaction_count",
            "total_amount", "status", "created_at",
        ]
        available_cols = [c for c in display_cols if c in filtered.columns]

        st.dataframe(
            filtered[available_cols].reset_index(drop=True),
            use_container_width=True,
            height=500,
        )

        # Export button
        if st.button("Export Alert Queue (CSV)"):
            csv_path = OUTPUTS_DIR / "alert_queue_export.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            filtered.to_csv(csv_path, index=False)
            st.success(f"Exported to {csv_path}")

    # ═══════════════════════════════════════════════════════════════════
    # TAB 2: Alert Detail
    # ═══════════════════════════════════════════════════════════════════
    with tab_detail:
        st.header("Alert Detail View")

        if len(filtered) == 0:
            st.warning("No alerts match the current filters.")
        else:
            alert_ids = filtered["alert_id"].tolist()
            selected_id = st.selectbox("Select Alert", alert_ids)

            if selected_id:
                alert_row = filtered[filtered["alert_id"] == selected_id].iloc[0]

                # Header
                severity_color = {
                    "HIGH": "red", "MEDIUM": "orange", "LOW": "green"
                }.get(alert_row["severity"], "gray")

                st.markdown(
                    f"### Alert: {alert_row['alert_id']} "
                    f"<span style='color:{severity_color};font-weight:bold;'>"
                    f"[{alert_row['severity']}]</span>",
                    unsafe_allow_html=True,
                )

                # Score breakdown
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Ensemble Score", f"{alert_row['ensemble_score']:.3f}")
                col2.metric("ML Score", f"{alert_row.get('ml_score', 0):.3f}")
                col3.metric("Graph Score", f"{alert_row.get('graph_score', 0):.3f}")
                col4.metric("Status", alert_row["status"])

                st.markdown("---")

                # Entity and transactions
                col_left, col_right = st.columns(2)

                with col_left:
                    st.subheader("Entity Information")
                    st.write(f"**Entity ID:** {alert_row['entity_id']}")
                    st.write(f"**Transaction Count:** {alert_row.get('transaction_count', 'N/A')}")
                    st.write(f"**Total Amount:** ${alert_row.get('total_amount', 0):,.2f}")
                    st.write(f"**Created:** {alert_row.get('created_at', 'N/A')}")

                    rules = alert_row.get("triggered_rules", [])
                    if isinstance(rules, str):
                        rules = json.loads(rules) if rules.startswith("[") else [rules]
                    st.write(f"**Triggered Rules:** {', '.join(rules) if rules else 'None'}")

                with col_right:
                    st.subheader("LLM Triage Summary")
                    summary = alert_row.get("llm_summary", "No LLM summary available.")
                    recommendation = alert_row.get("llm_recommendation", "N/A")
                    confidence = alert_row.get("confidence_level", "N/A")
                    typology = alert_row.get("typology", "N/A")

                    st.info(summary if summary else "No summary generated.")
                    st.write(f"**Recommended Disposition:** {recommendation}")
                    st.write(f"**Confidence Level:** {confidence}")
                    st.write(f"**Typology Classification:** {typology}")

                # Score gauge chart
                st.subheader("Risk Score Breakdown")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=alert_row["ensemble_score"],
                    title={"text": "Ensemble Risk Score"},
                    gauge={
                        "axis": {"range": [0, 1]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 0.5], "color": "lightgreen"},
                            {"range": [0.5, 0.7], "color": "yellow"},
                            {"range": [0.7, 1.0], "color": "red"},
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 4},
                            "thickness": 0.75,
                            "value": 0.5,
                        },
                    },
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════
    # TAB 3: Network Graph Visualization
    # ═══════════════════════════════════════════════════════════════════
    with tab_network:
        st.header("Transaction Network Visualization")
        st.markdown(
            "Interactive graph showing transaction flows between accounts. "
            "Node size reflects transaction volume; edges represent fund transfers."
        )

        n_nodes = st.slider("Number of nodes to display", 10, 100, 30, 5)

        # Build a sample network from alert data
        G = nx.DiGraph()
        rng = np.random.RandomState(42)

        entities = filtered["entity_id"].unique()[:n_nodes]
        all_nodes = list(entities)

        # Add additional counterparty nodes
        for entity in entities:
            n_counterparties = rng.randint(1, 5)
            for _ in range(n_counterparties):
                cp = f"CP_{rng.randint(10000, 99999)}"
                if cp not in all_nodes and len(all_nodes) < n_nodes * 2:
                    all_nodes.append(cp)
                    amount = float(rng.lognormal(9, 1.5))
                    G.add_edge(entity, cp, weight=amount)
                    if rng.random() > 0.6:
                        G.add_edge(cp, entity, weight=float(rng.lognormal(8, 1.5)))

        # Add some inter-entity connections
        for i in range(min(len(entities) - 1, 10)):
            if rng.random() > 0.5 and i + 1 < len(entities):
                G.add_edge(
                    entities[i], entities[i + 1],
                    weight=float(rng.lognormal(10, 1)),
                )

        if G.number_of_nodes() > 0:
            pos = nx.spring_layout(G, seed=42, k=2)

            # Edges
            edge_x, edge_y = [], []
            for u, v in G.edges():
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.8, color="#888"),
                hoverinfo="none",
                mode="lines",
                name="Transactions",
            )

            # Nodes
            node_x = [pos[n][0] for n in G.nodes()]
            node_y = [pos[n][1] for n in G.nodes()]
            node_text = list(G.nodes())

            # Color by whether node is in alerts
            entity_set = set(entities)
            node_colors = [
                "red" if n in entity_set else "lightblue"
                for n in G.nodes()
            ]
            node_sizes = [
                20 if n in entity_set else 10
                for n in G.nodes()
            ]

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode="markers+text",
                hoverinfo="text",
                text=[n[:15] for n in node_text],
                textposition="top center",
                textfont=dict(size=8),
                marker=dict(
                    color=node_colors,
                    size=node_sizes,
                    line_width=1,
                    line_color="black",
                ),
                name="Accounts",
            )

            fig_network = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title="Transaction Network (Red = Alerted Entities)",
                    showlegend=False,
                    hovermode="closest",
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=600,
                ),
            )
            st.plotly_chart(fig_network, use_container_width=True)

            # Network stats
            col1, col2, col3 = st.columns(3)
            col1.metric("Nodes", G.number_of_nodes())
            col2.metric("Edges", G.number_of_edges())
            col3.metric(
                "Density",
                f"{nx.density(G):.4f}",
            )
        else:
            st.info("No network data available to visualize.")

    # ═══════════════════════════════════════════════════════════════════
    # TAB 4: Alert Statistics
    # ═══════════════════════════════════════════════════════════════════
    with tab_stats:
        st.header("Alert Statistics")

        if len(alerts_df) == 0:
            st.warning("No alert data available.")
        else:
            # Severity distribution
            col1, col2 = st.columns(2)

            with col1:
                severity_counts = alerts_df["severity"].value_counts()
                fig_severity = px.pie(
                    values=severity_counts.values,
                    names=severity_counts.index,
                    title="Alerts by Severity",
                    color=severity_counts.index,
                    color_discrete_map={
                        "HIGH": "#dc3545",
                        "MEDIUM": "#ffc107",
                        "LOW": "#28a745",
                    },
                )
                st.plotly_chart(fig_severity, use_container_width=True)

            with col2:
                status_counts = alerts_df["status"].value_counts()
                fig_status = px.pie(
                    values=status_counts.values,
                    names=status_counts.index,
                    title="Alerts by Status",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                st.plotly_chart(fig_status, use_container_width=True)

            # Score distribution
            fig_scores = px.histogram(
                alerts_df,
                x="ensemble_score",
                color="severity",
                nbins=30,
                title="Ensemble Score Distribution by Severity",
                color_discrete_map={
                    "HIGH": "#dc3545",
                    "MEDIUM": "#ffc107",
                    "LOW": "#28a745",
                },
                barmode="overlay",
                opacity=0.7,
            )
            fig_scores.update_layout(xaxis_title="Ensemble Score", yaxis_title="Count")
            st.plotly_chart(fig_scores, use_container_width=True)

            # Rules triggered distribution
            if "triggered_rules" in alerts_df.columns:
                rule_counts: Dict[str, int] = defaultdict(int)
                for rules in alerts_df["triggered_rules"]:
                    if isinstance(rules, list):
                        for r in rules:
                            rule_counts[r] += 1
                    elif isinstance(rules, str) and rules.startswith("["):
                        for r in json.loads(rules):
                            rule_counts[r] += 1

                if rule_counts:
                    fig_rules = px.bar(
                        x=list(rule_counts.keys()),
                        y=list(rule_counts.values()),
                        title="Rules Triggered (Alert Count)",
                        labels={"x": "Rule", "y": "Alert Count"},
                        color=list(rule_counts.values()),
                        color_continuous_scale="Reds",
                    )
                    fig_rules.update_layout(showlegend=False)
                    st.plotly_chart(fig_rules, use_container_width=True)

            # LLM disposition distribution
            if "llm_recommendation" in alerts_df.columns:
                disp_counts = alerts_df["llm_recommendation"].value_counts()
                fig_disp = px.bar(
                    x=disp_counts.index,
                    y=disp_counts.values,
                    title="LLM Triage Disposition Recommendations",
                    labels={"x": "Disposition", "y": "Count"},
                    color=disp_counts.index,
                    color_discrete_map={
                        "ESCALATE_TO_SAR": "#dc3545",
                        "CONTINUE_MONITORING": "#ffc107",
                        "CLOSE": "#28a745",
                    },
                )
                st.plotly_chart(fig_disp, use_container_width=True)

            # Alerts over time
            if "created_at" in alerts_df.columns:
                alerts_df["created_date"] = pd.to_datetime(
                    alerts_df["created_at"], errors="coerce"
                ).dt.date
                daily_counts = alerts_df.groupby("created_date").size().reset_index(name="count")
                fig_timeline = px.line(
                    daily_counts,
                    x="created_date",
                    y="count",
                    title="Alert Volume Over Time",
                    labels={"created_date": "Date", "count": "Alerts"},
                )
                st.plotly_chart(fig_timeline, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════
    # TAB 5: Model Performance Monitoring
    # ═══════════════════════════════════════════════════════════════════
    with tab_performance:
        st.header("Model Performance Monitoring")

        metrics = load_model_metrics()

        # System comparison
        comparison = metrics.get("system_comparison", {}).get("system_metrics", {})
        if comparison:
            st.subheader("System Configuration Comparison")

            comparison_rows = []
            for sys_name, m in comparison.items():
                comparison_rows.append({
                    "System": sys_name.replace("_", " ").title(),
                    "AUC-ROC": m.get("auc_roc", 0),
                    "AUC-PR": m.get("auc_pr", 0),
                    "F1": m.get("f1", 0),
                    "Recall@5%FPR": m.get("recall_at_5pct_fpr", 0),
                })

            comp_df = pd.DataFrame(comparison_rows)
            st.dataframe(comp_df, use_container_width=True)

            # Bar chart comparison
            fig_comp = go.Figure()
            metric_names = ["AUC-ROC", "AUC-PR", "F1", "Recall@5%FPR"]
            for metric_name in metric_names:
                fig_comp.add_trace(go.Bar(
                    name=metric_name,
                    x=comp_df["System"],
                    y=comp_df[metric_name],
                ))
            fig_comp.update_layout(
                title="System Comparison: Key Metrics",
                barmode="group",
                yaxis_title="Score",
                xaxis_title="System Configuration",
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        # Best model metrics
        for key, model_metrics in metrics.items():
            if key.startswith("transaction_level_"):
                model_name = key.replace("transaction_level_", "")
                st.subheader(f"Best Model: {model_name}")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("AUC-ROC", f"{model_metrics.get('auc_roc', 0):.4f}")
                col2.metric("AUC-PR", f"{model_metrics.get('auc_pr', 0):.4f}")
                col3.metric("F1", f"{model_metrics.get('f1', 0):.4f}")
                col4.metric("Recall", f"{model_metrics.get('recall', 0):.4f}")

                col5, col6, col7, col8 = st.columns(4)
                col5.metric("Precision", f"{model_metrics.get('precision', 0):.4f}")
                col6.metric(
                    "Recall@1%FPR",
                    f"{model_metrics.get('recall_at_1pct_fpr', 0):.4f}",
                )
                col7.metric(
                    "Recall@5%FPR",
                    f"{model_metrics.get('recall_at_5pct_fpr', 0):.4f}",
                )
                col8.metric("Threshold", f"{model_metrics.get('threshold', 0.5):.4f}")

                # ROC curve (if available)
                roc_data = model_metrics.get("roc_curve")
                if roc_data:
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(
                        x=roc_data["fpr"],
                        y=roc_data["tpr"],
                        mode="lines",
                        name=f"ROC (AUC={model_metrics.get('auc_roc', 0):.3f})",
                        line=dict(color="blue", width=2),
                    ))
                    fig_roc.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode="lines",
                        name="Random",
                        line=dict(color="gray", width=1, dash="dash"),
                    ))
                    fig_roc.update_layout(
                        title="ROC Curve",
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate",
                        height=400,
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)

                # PR curve (if available)
                pr_data = model_metrics.get("pr_curve")
                if pr_data:
                    fig_pr = go.Figure()
                    fig_pr.add_trace(go.Scatter(
                        x=pr_data["recall"],
                        y=pr_data["precision"],
                        mode="lines",
                        name=f"PR (AUC={model_metrics.get('auc_pr', 0):.3f})",
                        line=dict(color="green", width=2),
                    ))
                    fig_pr.update_layout(
                        title="Precision-Recall Curve",
                        xaxis_title="Recall",
                        yaxis_title="Precision",
                        height=400,
                    )
                    st.plotly_chart(fig_pr, use_container_width=True)

                break  # Only show best model

        # Simulated model monitoring over time
        st.subheader("Model Performance Monitoring (Simulated)")
        dates = pd.date_range(end=datetime.utcnow(), periods=30, freq="D")
        rng = np.random.RandomState(42)
        monitoring_df = pd.DataFrame({
            "date": dates,
            "auc_roc": 0.96 + rng.normal(0, 0.005, 30).cumsum() * 0.001,
            "auc_pr": 0.55 + rng.normal(0, 0.008, 30).cumsum() * 0.001,
            "alert_volume": rng.poisson(25, 30),
            "false_positive_rate": 0.04 + rng.normal(0, 0.002, 30).cumsum() * 0.0005,
        })

        col1, col2 = st.columns(2)

        with col1:
            fig_auc = px.line(
                monitoring_df,
                x="date",
                y=["auc_roc", "auc_pr"],
                title="AUC Metrics Over Time",
                labels={"value": "Score", "date": "Date"},
            )
            fig_auc.update_layout(yaxis_range=[0.4, 1.0])
            st.plotly_chart(fig_auc, use_container_width=True)

        with col2:
            fig_fpr = px.line(
                monitoring_df,
                x="date",
                y="false_positive_rate",
                title="False Positive Rate Over Time",
                labels={"false_positive_rate": "FPR", "date": "Date"},
            )
            fig_fpr.add_hline(
                y=0.05,
                line_dash="dash",
                line_color="red",
                annotation_text="5% threshold",
            )
            st.plotly_chart(fig_fpr, use_container_width=True)

    # ─── Footer ──────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "AML Transaction Monitoring Dashboard | "
        "For internal compliance use only | "
        f"Last refreshed: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
    )


if __name__ == "__main__":
    main()
