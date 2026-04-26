"""
Compliance Reporting Module.

Generates regulatory-ready reports for AML compliance:
  - SAR (Suspicious Activity Report) draft narratives
  - Alert statistics reports (volume by type, severity, disposition)
  - Model performance reports (detection rate, false positive rate)
  - Regulatory metrics (SLA compliance, alert aging)
  - Export to HTML and structured formats
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.alert_generator import Alert
from src.config import Config

logger = logging.getLogger(__name__)

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"


class ComplianceReporter:
    """
    Generates compliance reports for AML regulatory requirements.

    Parameters
    ----------
    config : Config
        Master configuration object.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.report_dir = REPORT_DIR
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def generate_sar_narrative(self, alert: Alert, triage_result: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a SAR (Suspicious Activity Report) draft narrative.

        The narrative follows FinCEN SAR filing guidelines:
        - Who is conducting the suspicious activity?
        - What instruments or mechanisms are being used?
        - When did the activity occur?
        - Where did the activity take place?
        - Why does the activity appear suspicious?

        Parameters
        ----------
        alert : Alert
            The alert to generate a SAR narrative for.
        triage_result : Dict[str, Any], optional
            LLM triage output for enhanced narrative.

        Returns
        -------
        str
            Draft SAR narrative ready for analyst review.
        """
        transactions = alert.transactions
        context = alert.context

        # Compute aggregate transaction details
        amounts = [t.get("amount", 0) for t in transactions]
        total_amount = sum(amounts)
        currencies = list(set(t.get("payment_currency", "USD") for t in transactions))
        formats = list(set(t.get("payment_format", t.get("payment_format_normalized", "N/A")) for t in transactions))

        dates = [t.get("timestamp", t.get("date", "")) for t in transactions if t.get("timestamp") or t.get("date")]
        date_range = f"{min(dates)} to {max(dates)}" if dates else "Date range unavailable"

        counterparties = set()
        for t in transactions:
            counterparties.add(t.get("to_id", ""))
            counterparties.add(t.get("from_id", ""))
        counterparties.discard(alert.entity_id)
        counterparties.discard("")

        # Build the SAR narrative sections
        sections = []

        # WHO
        sections.append(
            f"Subject: Account {alert.entity_id}\n"
            f"This report concerns suspicious activity identified on account {alert.entity_id}. "
            f"The account was flagged by our automated monitoring system with a risk score of "
            f"{alert.ensemble_score:.3f} (severity: {alert.severity})."
        )

        # WHAT
        rules_desc = ", ".join(alert.triggered_rules) if alert.triggered_rules else "general pattern analysis"
        sections.append(
            f"\nActivity Type: The following suspicious patterns were detected: {rules_desc}. "
            f"A total of {len(transactions)} transactions were flagged, involving "
            f"{', '.join(formats)} payment method(s) in {', '.join(currencies)} currency."
        )

        # WHEN
        sections.append(
            f"\nTimeframe: The suspicious activity occurred during the period {date_range}. "
            f"The transactions include {len(transactions)} individual transaction(s) "
            f"totaling approximately ${total_amount:,.2f}."
        )

        # WHERE
        if counterparties:
            cp_list = list(counterparties)[:10]
            sections.append(
                f"\nCounterparties: The subject transacted with {len(counterparties)} unique "
                f"counterparties. Key counterparties include: {', '.join(cp_list)}."
            )

        # WHY
        why_reasons = []
        if "structuring" in alert.triggered_rules:
            why_reasons.append(
                "multiple transactions structured just below the $10,000 Currency Transaction "
                "Report (CTR) threshold, suggesting potential willful evasion of reporting requirements"
            )
        if "rapid_movement" in alert.triggered_rules:
            why_reasons.append(
                "rapid inflow-outflow patterns where received funds were quickly transferred "
                "out, characteristic of layering activity"
            )
        if "round_tripping" in alert.triggered_rules:
            why_reasons.append(
                "circular fund flow patterns where money returned to accounts associated "
                "with the originator through intermediaries"
            )
        if "dormant_reactivation" in alert.triggered_rules:
            why_reasons.append(
                "sudden reactivation of a previously dormant account with significant "
                "transaction volume"
            )
        if "velocity" in alert.triggered_rules:
            why_reasons.append(
                "abnormally high transaction frequency exceeding the account's historical "
                "pattern"
            )
        if "geographic_risk" in alert.triggered_rules:
            why_reasons.append(
                "transactions involving jurisdictions identified as high-risk by FATF"
            )

        if not why_reasons:
            why_reasons.append(
                "the combination of ML model scoring, graph network analysis, and "
                "behavioral pattern analysis indicates activity inconsistent with "
                "legitimate financial activity"
            )

        sections.append(
            f"\nSuspicious Indicators: This activity is deemed suspicious because of "
            + "; ".join(why_reasons) + "."
        )

        # Add LLM triage narrative if available
        if triage_result and triage_result.get("narrative_summary"):
            sections.append(
                f"\nAI-Assisted Analysis: {triage_result['narrative_summary']}"
            )
            if triage_result.get("disposition_rationale"):
                sections.append(
                    f"\nAssessment: {triage_result['disposition_rationale']}"
                )

        # Transaction details
        sections.append("\nTransaction Details:")
        for i, txn in enumerate(transactions[:20], 1):
            sections.append(
                f"  {i}. Date: {txn.get('timestamp', txn.get('date', 'N/A'))} | "
                f"From: {txn.get('from_id', 'N/A')} | "
                f"To: {txn.get('to_id', 'N/A')} | "
                f"Amount: ${txn.get('amount', 0):,.2f} | "
                f"Format: {txn.get('payment_format', txn.get('payment_format_normalized', 'N/A'))}"
            )
        if len(transactions) > 20:
            sections.append(f"  ... and {len(transactions) - 20} additional transactions")

        # Disclaimer
        sections.append(
            f"\n--- DRAFT: Generated on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC ---\n"
            f"This narrative was generated by an automated system and requires analyst review "
            f"before SAR filing. Alert ID: {alert.alert_id}"
        )

        return "\n".join(sections)

    def generate_alert_statistics_report(
        self, alerts: List[Alert], triage_results: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Generate alert volume and distribution statistics.

        Parameters
        ----------
        alerts : List[Alert]
            All generated alerts.
        triage_results : List[Dict], optional
            LLM triage results.

        Returns
        -------
        Dict[str, Any]
            Structured statistics report.
        """
        if not alerts:
            return {"total_alerts": 0, "generated_at": datetime.utcnow().isoformat()}

        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "reporting_period": {
                "start": min(a.created_at for a in alerts).isoformat(),
                "end": max(a.created_at for a in alerts).isoformat(),
            },
            "total_alerts": len(alerts),
            "by_severity": {
                "HIGH": sum(1 for a in alerts if a.severity == "HIGH"),
                "MEDIUM": sum(1 for a in alerts if a.severity == "MEDIUM"),
                "LOW": sum(1 for a in alerts if a.severity == "LOW"),
            },
            "by_status": {
                "OPEN": sum(1 for a in alerts if a.status == "OPEN"),
                "IN_REVIEW": sum(1 for a in alerts if a.status == "IN_REVIEW"),
                "ESCALATED": sum(1 for a in alerts if a.status == "ESCALATED"),
                "CLOSED": sum(1 for a in alerts if a.status == "CLOSED"),
            },
            "score_statistics": {
                "mean": float(np.mean([a.ensemble_score for a in alerts])),
                "median": float(np.median([a.ensemble_score for a in alerts])),
                "std": float(np.std([a.ensemble_score for a in alerts])),
                "p90": float(np.percentile([a.ensemble_score for a in alerts], 90)),
                "p95": float(np.percentile([a.ensemble_score for a in alerts], 95)),
            },
            "rules_triggered": {},
            "top_entities": [],
        }

        # Rules triggered distribution
        all_rules = [r for a in alerts for r in a.triggered_rules]
        for rule in set(all_rules):
            report["rules_triggered"][rule] = all_rules.count(rule)

        # Top entities by alert count
        entity_counts: Dict[str, int] = {}
        for a in alerts:
            entity_counts[a.entity_id] = entity_counts.get(a.entity_id, 0) + 1
        report["top_entities"] = sorted(
            [{"entity_id": k, "alert_count": v} for k, v in entity_counts.items()],
            key=lambda x: x["alert_count"],
            reverse=True,
        )[:20]

        # SLA compliance
        now = datetime.utcnow()
        breached = sum(1 for a in alerts if a.status == "OPEN" and a.sla_deadline < now)
        report["sla_compliance"] = {
            "total_open": sum(1 for a in alerts if a.status == "OPEN"),
            "sla_breached": breached,
            "breach_rate_pct": float(breached / len(alerts) * 100) if alerts else 0,
        }

        # LLM triage statistics
        if triage_results:
            dispositions = [r.get("recommended_disposition", "N/A") for r in triage_results]
            report["llm_triage"] = {
                "total_triaged": len(triage_results),
                "disposition_distribution": {d: dispositions.count(d) for d in set(dispositions)},
                "sar_recommendation_rate": float(
                    dispositions.count("ESCALATE_TO_SAR") / len(dispositions) * 100
                ) if dispositions else 0,
            }

        return report

    def generate_model_performance_report(
        self, metrics: Dict[str, Any], comparison: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, Any]:
        """
        Generate a model performance report for regulatory review.

        Parameters
        ----------
        metrics : Dict[str, Any]
            Model evaluation metrics.
        comparison : Dict[str, Dict], optional
            Comparison metrics for different model configurations.

        Returns
        -------
        Dict[str, Any]
            Model performance report.
        """
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "model_performance": metrics,
            "regulatory_metrics": {
                "detection_rate": metrics.get("recall", 0),
                "false_positive_rate": metrics.get("false_positive_rate", 0),
                "alert_to_sar_conversion": metrics.get("alert_to_sar_rate", 0),
                "model_auc_roc": metrics.get("auc_roc", 0),
                "model_auc_pr": metrics.get("auc_pr", 0),
            },
        }

        if comparison:
            report["model_comparison"] = comparison

        return report

    def generate_regulatory_metrics_report(
        self, alerts: List[Alert]
    ) -> Dict[str, Any]:
        """
        Generate regulatory metrics including SLA compliance and alert aging.

        Parameters
        ----------
        alerts : List[Alert]
            All alerts for the reporting period.

        Returns
        -------
        Dict[str, Any]
            Regulatory metrics report.
        """
        now = datetime.utcnow()

        open_alerts = [a for a in alerts if a.status == "OPEN"]
        aging_hours = [
            (now - a.created_at).total_seconds() / 3600 for a in open_alerts
        ]

        return {
            "generated_at": now.isoformat(),
            "alert_aging": {
                "open_alerts": len(open_alerts),
                "mean_age_hours": float(np.mean(aging_hours)) if aging_hours else 0,
                "median_age_hours": float(np.median(aging_hours)) if aging_hours else 0,
                "max_age_hours": float(np.max(aging_hours)) if aging_hours else 0,
                "aged_over_24h": sum(1 for h in aging_hours if h > 24),
                "aged_over_72h": sum(1 for h in aging_hours if h > 72),
                "aged_over_7d": sum(1 for h in aging_hours if h > 168),
            },
            "sla_compliance": {
                "high_sla_hours": self.config.alert.high_severity_sla_hours,
                "medium_sla_hours": self.config.alert.medium_severity_sla_hours,
                "low_sla_hours": self.config.alert.low_severity_sla_hours,
                "high_breached": sum(
                    1 for a in alerts
                    if a.severity == "HIGH" and a.status == "OPEN" and a.sla_deadline < now
                ),
                "medium_breached": sum(
                    1 for a in alerts
                    if a.severity == "MEDIUM" and a.status == "OPEN" and a.sla_deadline < now
                ),
                "low_breached": sum(
                    1 for a in alerts
                    if a.severity == "LOW" and a.status == "OPEN" and a.sla_deadline < now
                ),
            },
            "throughput": {
                "total_alerts": len(alerts),
                "closed_alerts": sum(1 for a in alerts if a.status == "CLOSED"),
                "escalated_alerts": sum(1 for a in alerts if a.status == "ESCALATED"),
                "closure_rate_pct": float(
                    sum(1 for a in alerts if a.status == "CLOSED") / len(alerts) * 100
                ) if alerts else 0,
                "escalation_rate_pct": float(
                    sum(1 for a in alerts if a.status == "ESCALATED") / len(alerts) * 100
                ) if alerts else 0,
            },
        }

    def export_report_html(
        self,
        report: Dict[str, Any],
        title: str = "AML Compliance Report",
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Export a report as an HTML file.

        Parameters
        ----------
        report : Dict[str, Any]
            Report data.
        title : str
            Report title.
        output_path : Path, optional
            Output file path.

        Returns
        -------
        Path
            Path to the generated HTML file.
        """
        output_path = output_path or self.report_dir / f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        html = self._build_html_report(report, title)

        output_path.write_text(html, encoding="utf-8")
        logger.info(f"Report exported to {output_path}")

        return output_path

    def _build_html_report(self, report: Dict[str, Any], title: str) -> str:
        """Build an HTML string from report data."""
        timestamp = report.get("generated_at", datetime.utcnow().isoformat())

        html_parts = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            f"  <title>{title}</title>",
            "  <meta charset='utf-8'>",
            "  <style>",
            "    body { font-family: 'Segoe UI', Tahoma, sans-serif; margin: 40px; color: #333; }",
            "    h1 { color: #1a1a2e; border-bottom: 3px solid #16213e; padding-bottom: 10px; }",
            "    h2 { color: #16213e; margin-top: 30px; }",
            "    table { border-collapse: collapse; width: 100%; margin: 15px 0; }",
            "    th, td { border: 1px solid #ddd; padding: 10px; text-align: left; }",
            "    th { background-color: #16213e; color: white; }",
            "    tr:nth-child(even) { background-color: #f8f9fa; }",
            "    .metric-card { display: inline-block; background: #f8f9fa; border-radius: 8px; ",
            "      padding: 15px 25px; margin: 8px; border-left: 4px solid #16213e; }",
            "    .metric-value { font-size: 24px; font-weight: bold; color: #16213e; }",
            "    .metric-label { font-size: 12px; color: #666; text-transform: uppercase; }",
            "    .severity-high { color: #dc3545; font-weight: bold; }",
            "    .severity-medium { color: #ffc107; font-weight: bold; }",
            "    .severity-low { color: #28a745; font-weight: bold; }",
            "    .footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; ",
            "      font-size: 12px; color: #999; }",
            "  </style>",
            "</head>",
            "<body>",
            f"  <h1>{title}</h1>",
            f"  <p>Generated: {timestamp}</p>",
        ]

        # Render report sections
        for section_name, section_data in report.items():
            if section_name in ("generated_at",):
                continue

            html_parts.append(f"  <h2>{section_name.replace('_', ' ').title()}</h2>")

            if isinstance(section_data, dict):
                html_parts.append("  <table>")
                html_parts.append("    <tr><th>Metric</th><th>Value</th></tr>")
                for key, value in section_data.items():
                    if isinstance(value, dict):
                        for subkey, subval in value.items():
                            display_val = f"{subval:,.2f}" if isinstance(subval, float) else str(subval)
                            html_parts.append(f"    <tr><td>{key} - {subkey}</td><td>{display_val}</td></tr>")
                    elif isinstance(value, list):
                        html_parts.append(f"    <tr><td>{key}</td><td>{len(value)} items</td></tr>")
                    else:
                        display_val = f"{value:,.2f}" if isinstance(value, float) else str(value)
                        html_parts.append(f"    <tr><td>{key}</td><td>{display_val}</td></tr>")
                html_parts.append("  </table>")
            else:
                display_val = f"{section_data:,.2f}" if isinstance(section_data, float) else str(section_data)
                html_parts.append(f"  <div class='metric-card'>")
                html_parts.append(f"    <div class='metric-value'>{display_val}</div>")
                html_parts.append(f"    <div class='metric-label'>{section_name}</div>")
                html_parts.append(f"  </div>")

        html_parts.extend([
            "  <div class='footer'>",
            "    <p>This report was generated by the AML Transaction Monitoring System. ",
            "    It is intended for internal compliance use only and does not constitute legal advice.</p>",
            "  </div>",
            "</body>",
            "</html>",
        ])

        return "\n".join(html_parts)

    def generate_full_compliance_package(
        self,
        alerts: List[Alert],
        triage_results: Optional[List[Dict]] = None,
        model_metrics: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Generate a full compliance reporting package.

        Includes all report types in a single structured output.

        Parameters
        ----------
        alerts : List[Alert]
            All alerts.
        triage_results : List[Dict], optional
            LLM triage results.
        model_metrics : Dict, optional
            Model performance metrics.

        Returns
        -------
        Dict[str, Any]
            Complete compliance package.
        """
        logger.info("Generating full compliance reporting package")

        package = {
            "generated_at": datetime.utcnow().isoformat(),
            "alert_statistics": self.generate_alert_statistics_report(alerts, triage_results),
            "regulatory_metrics": self.generate_regulatory_metrics_report(alerts),
        }

        if model_metrics:
            package["model_performance"] = self.generate_model_performance_report(model_metrics)

        # Generate SAR narratives for HIGH severity alerts
        high_alerts = [a for a in alerts if a.severity == "HIGH"]
        sar_drafts = []
        for alert in high_alerts[:50]:  # Limit for performance
            triage = None
            if triage_results:
                matching = [r for r in triage_results if r.get("alert_id") == alert.alert_id]
                triage = matching[0] if matching else None

            sar_drafts.append({
                "alert_id": alert.alert_id,
                "entity_id": alert.entity_id,
                "severity": alert.severity,
                "narrative": self.generate_sar_narrative(alert, triage),
            })

        package["sar_drafts"] = sar_drafts
        package["sar_draft_count"] = len(sar_drafts)

        # Export HTML report
        html_path = self.export_report_html(
            package["alert_statistics"],
            title="AML Alert Statistics Report",
        )
        package["html_report_path"] = str(html_path)

        logger.info(
            f"Compliance package generated: "
            f"{len(alerts)} alerts, {len(sar_drafts)} SAR drafts, "
            f"HTML report at {html_path}"
        )

        return package
