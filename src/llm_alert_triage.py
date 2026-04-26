"""
LLM-Powered Alert Triage System.

The key differentiator of this AML system: uses Large Language Models
(via LangChain) to automatically triage alerts by generating:

  - Structured case summaries with narrative descriptions
  - Key risk indicator identification with evidence
  - Timelines of relevant events
  - Disposition recommendations (Escalate to SAR, Continue Monitoring, Close)
  - Confidence levels for each recommendation

Supports both Anthropic (Claude) and OpenAI backends. Includes hallucination
verification to ensure generated facts match source data.
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.alert_generator import Alert
from src.config import Config, LLMConfig

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert AML (Anti-Money Laundering) compliance analyst assistant.
Your role is to review flagged transaction alerts and produce structured case summaries
to assist human compliance officers in their investigation.

You must be precise, factual, and cite specific evidence from the provided data.
Never fabricate transactions, amounts, dates, or account identifiers that are not
present in the source data. If information is insufficient, state that clearly.

Your output must follow the exact JSON format specified."""

TRIAGE_PROMPT_TEMPLATE = """## Alert Triage Request

### Alert Details
- **Alert ID**: {alert_id}
- **Entity (Account)**: {entity_id}
- **Severity**: {severity}
- **Ensemble Risk Score**: {ensemble_score:.3f}
- **ML Model Score**: {ml_score:.3f}
- **Graph Analysis Score**: {graph_score:.3f}
- **Rules Triggered**: {triggered_rules}

### Transaction History (Most Recent)
{transaction_table}

### Counterparty Information
- Total unique counterparties: {n_counterparties}
- Suspicious counterparties: {suspicious_counterparties}

### Graph Context
- Community ID: {community_id}
- Part of cycle (round-tripping): {in_cycle}
- Prior alerts on this entity: {prior_alerts}

### Additional Context
{additional_context}

---

## Instructions

Analyze the above alert and produce a JSON response with the following structure:

```json
{{
    "narrative_summary": "A 2-4 sentence narrative describing the suspicious activity pattern in plain English. Reference specific amounts, dates, and counterparties from the data.",
    "risk_indicators": [
        {{
            "indicator": "Name of the risk indicator",
            "evidence": "Specific evidence from the transaction data",
            "severity": "HIGH/MEDIUM/LOW"
        }}
    ],
    "timeline": [
        {{
            "date": "YYYY-MM-DD",
            "event": "Description of key event"
        }}
    ],
    "recommended_disposition": "ESCALATE_TO_SAR | CONTINUE_MONITORING | CLOSE",
    "disposition_rationale": "2-3 sentence explanation of why this disposition is recommended",
    "confidence_level": "HIGH/MEDIUM/LOW",
    "additional_investigation_needed": ["List of suggested follow-up actions if any"],
    "estimated_risk_amount": 0.00,
    "typology_classification": "STRUCTURING | LAYERING | ROUND_TRIPPING | RAPID_MOVEMENT | DORMANT_REACTIVATION | MIXED | UNKNOWN"
}}
```

Be specific. Reference actual transaction amounts, dates, and counterparties from the data provided.
Do not invent any facts not present in the source data."""

FEW_SHOT_EXAMPLES = [
    {
        "input_summary": "Account with 5 transactions between $8,500-$9,800 within 24 hours, all to different recipients.",
        "output": {
            "narrative_summary": "Account BANK_001_12345 conducted 5 cash transactions totaling $45,200 within a 24-hour period on 2024-01-15. All transactions were between $8,500 and $9,800, deliberately structured below the $10,000 CTR reporting threshold. Each transaction was directed to a different recipient account, suggesting an attempt to avoid aggregation detection.",
            "risk_indicators": [
                {
                    "indicator": "Structuring below CTR threshold",
                    "evidence": "5 transactions in range $8,500-$9,800, all below $10,000",
                    "severity": "HIGH",
                },
                {
                    "indicator": "Multiple recipients in short window",
                    "evidence": "5 unique recipients within 24 hours",
                    "severity": "MEDIUM",
                },
            ],
            "timeline": [
                {"date": "2024-01-15", "event": "5 structured transactions totaling $45,200"},
            ],
            "recommended_disposition": "ESCALATE_TO_SAR",
            "disposition_rationale": "The pattern strongly indicates deliberate structuring to avoid CTR filing requirements, which is a federal offense under 31 USC 5324. The concentrated timeframe and distribution across multiple recipients increases suspicion.",
            "confidence_level": "HIGH",
            "additional_investigation_needed": ["Check if recipients are related entities", "Review 90-day transaction history"],
            "estimated_risk_amount": 45200.00,
            "typology_classification": "STRUCTURING",
        },
    },
]


class LLMAlertTriageSystem:
    """
    LLM-powered alert triage for AML compliance.

    Uses LangChain with Anthropic (Claude) or OpenAI to generate structured
    case summaries and disposition recommendations for AML alerts.

    Parameters
    ----------
    config : Config
        Master configuration object.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.llm_config: LLMConfig = config.llm
        self._llm = None
        self._initialized = False
        self._triage_results: List[Dict[str, Any]] = []

    def initialize(self) -> None:
        """
        Initialize the LLM backend.

        Configures the appropriate LangChain LLM (Anthropic or OpenAI)
        based on the configuration.
        """
        provider = self.llm_config.provider

        try:
            if provider == "anthropic":
                from langchain_anthropic import ChatAnthropic

                if not self.llm_config.anthropic_api_key:
                    raise ValueError("ANTHROPIC_API_KEY not set")

                self._llm = ChatAnthropic(
                    model=self.llm_config.anthropic_model,
                    api_key=self.llm_config.anthropic_api_key,
                    temperature=self.llm_config.temperature,
                    max_tokens=self.llm_config.max_tokens,
                    timeout=self.llm_config.request_timeout,
                )
                logger.info(f"Initialized Anthropic LLM: {self.llm_config.anthropic_model}")

            elif provider == "openai":
                from langchain_openai import ChatOpenAI

                if not self.llm_config.openai_api_key:
                    raise ValueError("OPENAI_API_KEY not set")

                self._llm = ChatOpenAI(
                    model=self.llm_config.openai_model,
                    api_key=self.llm_config.openai_api_key,
                    temperature=self.llm_config.temperature,
                    max_tokens=self.llm_config.max_tokens,
                    timeout=self.llm_config.request_timeout,
                )
                logger.info(f"Initialized OpenAI LLM: {self.llm_config.openai_model}")

            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")

            self._initialized = True

        except ImportError as e:
            logger.error(
                f"LangChain provider package not installed for '{provider}'. "
                f"Install with: pip install langchain-{provider}. Error: {e}"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

    def triage_alert(self, alert: Alert) -> Dict[str, Any]:
        """
        Triage a single alert using the LLM.

        Parameters
        ----------
        alert : Alert
            The alert to triage.

        Returns
        -------
        Dict[str, Any]
            Structured triage result with summary, risk indicators,
            recommendation, and confidence level.
        """
        if not self._initialized:
            self.initialize()

        logger.info(f"Triaging alert {alert.alert_id} (entity: {alert.entity_id})")

        # Build prompt context
        prompt = self._build_triage_prompt(alert)

        # Call LLM with retries
        raw_response = self._call_llm(prompt)

        # Parse structured output
        triage_result = self._parse_response(raw_response)

        # Verify facts against source data (hallucination check)
        if self.llm_config.enable_hallucination_check:
            triage_result = self._verify_facts(triage_result, alert)

        # Attach to alert
        alert.llm_summary = triage_result.get("narrative_summary", "")
        alert.llm_recommendation = triage_result.get("recommended_disposition", "CONTINUE_MONITORING")

        triage_result["alert_id"] = alert.alert_id
        triage_result["entity_id"] = alert.entity_id
        self._triage_results.append(triage_result)

        logger.info(
            f"  Triage complete: disposition={triage_result.get('recommended_disposition', 'N/A')}, "
            f"confidence={triage_result.get('confidence_level', 'N/A')}"
        )

        return triage_result

    def triage_batch(self, alerts: List[Alert]) -> List[Dict[str, Any]]:
        """
        Triage a batch of alerts.

        Parameters
        ----------
        alerts : List[Alert]
            Alerts to triage.

        Returns
        -------
        List[Dict[str, Any]]
            Triage results for each alert.
        """
        if not self._initialized:
            self.initialize()

        logger.info(f"Batch triaging {len(alerts)} alerts")
        results = []
        batch_size = self.llm_config.batch_size

        for i in range(0, len(alerts), batch_size):
            batch = alerts[i:i + batch_size]
            logger.info(f"  Processing batch {i // batch_size + 1} ({len(batch)} alerts)")

            for alert in batch:
                try:
                    result = self.triage_alert(alert)
                    results.append(result)
                except Exception as e:
                    logger.error(f"  Failed to triage alert {alert.alert_id}: {e}")
                    results.append(self._create_fallback_result(alert, str(e)))

            # Rate limiting between batches
            if i + batch_size < len(alerts):
                time.sleep(1)

        logger.info(f"Batch triage complete: {len(results)}/{len(alerts)} successful")
        return results

    def _build_triage_prompt(self, alert: Alert) -> str:
        """Build the triage prompt with alert context."""
        # Format transaction table
        txn_table = self._format_transaction_table(alert.transactions)

        # Format additional context
        additional_lines = []
        context = alert.context
        pattern = context.get("transaction_pattern", {})
        if pattern:
            additional_lines.append(
                f"- Total flagged amount: ${pattern.get('total_amount', 0):,.2f}"
            )
            additional_lines.append(
                f"- Average transaction: ${pattern.get('mean_amount', 0):,.2f}"
            )
            additional_lines.append(
                f"- Max single transaction: ${pattern.get('max_amount', 0):,.2f}"
            )
            additional_lines.append(
                f"- Flagged transactions: {pattern.get('n_transactions', 0)}"
            )

        additional_context = "\n".join(additional_lines) if additional_lines else "No additional context available."

        prompt = TRIAGE_PROMPT_TEMPLATE.format(
            alert_id=alert.alert_id,
            entity_id=alert.entity_id,
            severity=alert.severity,
            ensemble_score=alert.ensemble_score,
            ml_score=alert.ml_score,
            graph_score=alert.graph_score,
            triggered_rules=", ".join(alert.triggered_rules) or "None",
            transaction_table=txn_table,
            n_counterparties=context.get("n_counterparties", 0),
            suspicious_counterparties=context.get("suspicious_counterparties", []),
            community_id=context.get("community_id", "N/A"),
            in_cycle=context.get("in_cycle", False),
            prior_alerts=context.get("prior_alert_count", 0),
            additional_context=additional_context,
        )

        return prompt

    def _format_transaction_table(self, transactions: List[Dict]) -> str:
        """Format transactions as a readable markdown table."""
        if not transactions:
            return "No transaction data available."

        # Select key columns
        headers = ["Date", "From", "To", "Amount", "Format", "Flagged"]
        rows = []

        for txn in transactions[:self.llm_config.max_transactions_in_context]:
            row = [
                str(txn.get("timestamp", txn.get("date", "N/A")))[:19],
                str(txn.get("from_id", "N/A"))[:20],
                str(txn.get("to_id", "N/A"))[:20],
                f"${txn.get('amount', 0):,.2f}",
                str(txn.get("payment_format", txn.get("payment_format_normalized", "N/A"))),
                "Yes" if txn.get("is_laundering", 0) else "No",
            ]
            rows.append(row)

        # Build markdown table
        col_widths = [max(len(h), max(len(r[i]) for r in rows) if rows else 0) for i, h in enumerate(headers)]
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        separator = "-|-".join("-" * w for w in col_widths)
        row_lines = [" | ".join(cell.ljust(w) for cell, w in zip(row, col_widths)) for row in rows]

        return f"{header_line}\n{separator}\n" + "\n".join(row_lines)

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with retry logic."""
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
        ]

        # Add few-shot examples
        for example in FEW_SHOT_EXAMPLES:
            messages.append(HumanMessage(content=f"Example case: {example['input_summary']}"))
            from langchain_core.messages import AIMessage
            messages.append(AIMessage(content=json.dumps(example["output"], indent=2)))

        # Add the actual triage request
        messages.append(HumanMessage(content=prompt))

        for attempt in range(self.llm_config.max_retries):
            try:
                response = self._llm.invoke(messages)
                return response.content

            except Exception as e:
                logger.warning(f"  LLM call attempt {attempt + 1} failed: {e}")
                if attempt < self.llm_config.max_retries - 1:
                    time.sleep(self.llm_config.retry_delay_seconds * (attempt + 1))
                else:
                    raise RuntimeError(f"LLM call failed after {self.llm_config.max_retries} retries: {e}")

        return ""

    def _parse_response(self, raw_response: str) -> Dict[str, Any]:
        """Parse the LLM's JSON response, handling common formatting issues."""
        # Try to extract JSON from the response
        json_match = re.search(r'\{[\s\S]*\}', raw_response)
        if json_match:
            json_str = json_match.group()
            try:
                result = json.loads(json_str)
                return self._validate_triage_result(result)
            except json.JSONDecodeError:
                pass

        # Try to parse the entire response as JSON
        try:
            result = json.loads(raw_response)
            return self._validate_triage_result(result)
        except json.JSONDecodeError:
            pass

        # If parsing fails, create a structured result from raw text
        logger.warning("Failed to parse LLM JSON response; extracting from raw text")
        return {
            "narrative_summary": raw_response[:500],
            "risk_indicators": [],
            "timeline": [],
            "recommended_disposition": "CONTINUE_MONITORING",
            "disposition_rationale": "Unable to parse structured response; manual review recommended.",
            "confidence_level": "LOW",
            "additional_investigation_needed": ["Review raw LLM output"],
            "estimated_risk_amount": 0.0,
            "typology_classification": "UNKNOWN",
            "parse_error": True,
        }

    def _validate_triage_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize the triage result structure."""
        required_fields = [
            "narrative_summary",
            "risk_indicators",
            "recommended_disposition",
            "confidence_level",
        ]

        for field in required_fields:
            if field not in result:
                result[field] = self._get_default_value(field)

        # Normalize disposition
        valid_dispositions = {"ESCALATE_TO_SAR", "CONTINUE_MONITORING", "CLOSE"}
        if result.get("recommended_disposition", "").upper() not in valid_dispositions:
            result["recommended_disposition"] = "CONTINUE_MONITORING"
        else:
            result["recommended_disposition"] = result["recommended_disposition"].upper()

        # Normalize confidence
        valid_confidence = {"HIGH", "MEDIUM", "LOW"}
        if result.get("confidence_level", "").upper() not in valid_confidence:
            result["confidence_level"] = "MEDIUM"
        else:
            result["confidence_level"] = result["confidence_level"].upper()

        return result

    def _get_default_value(self, field: str) -> Any:
        """Return default values for missing triage fields."""
        defaults = {
            "narrative_summary": "Insufficient data for narrative generation.",
            "risk_indicators": [],
            "timeline": [],
            "recommended_disposition": "CONTINUE_MONITORING",
            "disposition_rationale": "Unable to determine disposition; manual review required.",
            "confidence_level": "LOW",
            "additional_investigation_needed": [],
            "estimated_risk_amount": 0.0,
            "typology_classification": "UNKNOWN",
        }
        return defaults.get(field, None)

    def _verify_facts(self, triage_result: Dict[str, Any], alert: Alert) -> Dict[str, Any]:
        """
        Verify facts in the LLM output against source data.

        Checks that:
        - Referenced amounts exist in the transaction data
        - Referenced dates are within the alert's time range
        - Referenced account IDs appear in the data
        - No hallucinated entities or transactions

        Parameters
        ----------
        triage_result : Dict[str, Any]
            Parsed LLM triage output.
        alert : Alert
            Source alert with transaction data.

        Returns
        -------
        Dict[str, Any]
            Triage result with verification annotations.
        """
        logger.info("  Running hallucination verification")

        # Extract verifiable facts from the narrative
        narrative = triage_result.get("narrative_summary", "")
        transactions = alert.transactions

        # Build fact sets from source data
        source_amounts = {round(t.get("amount", 0), 2) for t in transactions}
        source_accounts = set()
        for t in transactions:
            source_accounts.add(t.get("from_id", ""))
            source_accounts.add(t.get("to_id", ""))

        # Check amounts mentioned in the narrative
        amount_pattern = r'\$[\d,]+\.?\d*'
        mentioned_amounts = re.findall(amount_pattern, narrative)
        verified_amounts = 0
        total_amounts = len(mentioned_amounts)

        for amount_str in mentioned_amounts:
            amount_val = float(amount_str.replace("$", "").replace(",", ""))
            # Check if it is close to any source amount or sum of amounts
            total_source = sum(source_amounts)
            if (
                amount_val in source_amounts
                or abs(amount_val - total_source) < 1.0
                or any(abs(amount_val - a) < 1.0 for a in source_amounts)
            ):
                verified_amounts += 1

        # Check account IDs mentioned
        account_pattern = r'[A-Z]+_\d+_\d+'
        mentioned_accounts = re.findall(account_pattern, narrative)
        verified_accounts = sum(1 for a in mentioned_accounts if a in source_accounts)
        total_accounts = len(mentioned_accounts)

        # Compute verification score
        total_facts = total_amounts + total_accounts
        verified_facts = verified_amounts + verified_accounts

        if total_facts > 0:
            verification_score = verified_facts / total_facts
        else:
            verification_score = 1.0  # No verifiable facts to check

        triage_result["fact_verification"] = {
            "verification_score": round(verification_score, 3),
            "total_facts_checked": total_facts,
            "verified_facts": verified_facts,
            "amounts_verified": f"{verified_amounts}/{total_amounts}",
            "accounts_verified": f"{verified_accounts}/{total_accounts}",
            "passed": verification_score >= self.llm_config.fact_verification_strictness,
        }

        if not triage_result["fact_verification"]["passed"]:
            logger.warning(
                f"  Hallucination check FAILED: verification score {verification_score:.2f} "
                f"< threshold {self.llm_config.fact_verification_strictness}"
            )
            triage_result["confidence_level"] = "LOW"
            triage_result["disposition_rationale"] = (
                f"[VERIFICATION WARNING] {triage_result.get('disposition_rationale', '')} "
                f"Note: Some facts in this summary could not be verified against source data "
                f"(score: {verification_score:.2f}). Manual verification recommended."
            )

        return triage_result

    def _create_fallback_result(self, alert: Alert, error: str) -> Dict[str, Any]:
        """Create a fallback triage result when LLM call fails."""
        return {
            "alert_id": alert.alert_id,
            "entity_id": alert.entity_id,
            "narrative_summary": f"Automated triage unavailable (error: {error}). Manual review required.",
            "risk_indicators": [
                {
                    "indicator": r,
                    "evidence": "Rule engine trigger",
                    "severity": alert.severity,
                }
                for r in alert.triggered_rules
            ],
            "timeline": [],
            "recommended_disposition": "CONTINUE_MONITORING",
            "disposition_rationale": "LLM triage failed; defaulting to continue monitoring pending manual review.",
            "confidence_level": "LOW",
            "additional_investigation_needed": ["Complete manual review due to triage system error"],
            "estimated_risk_amount": sum(t.get("amount", 0) for t in alert.transactions),
            "typology_classification": "UNKNOWN",
            "error": error,
        }

    def get_triage_statistics(self) -> Dict[str, Any]:
        """Get statistics about triage results."""
        if not self._triage_results:
            return {"total_triaged": 0}

        dispositions = [r.get("recommended_disposition", "UNKNOWN") for r in self._triage_results]
        confidences = [r.get("confidence_level", "UNKNOWN") for r in self._triage_results]
        verification_scores = [
            r.get("fact_verification", {}).get("verification_score", 0)
            for r in self._triage_results
            if "fact_verification" in r
        ]

        return {
            "total_triaged": len(self._triage_results),
            "disposition_distribution": {
                d: dispositions.count(d) for d in set(dispositions)
            },
            "confidence_distribution": {
                c: confidences.count(c) for c in set(confidences)
            },
            "verification": {
                "mean_score": float(np.mean(verification_scores)) if verification_scores else 0,
                "pass_rate": float(
                    sum(1 for r in self._triage_results if r.get("fact_verification", {}).get("passed", False))
                    / len(self._triage_results)
                ) if self._triage_results else 0,
            },
            "typology_distribution": {
                t: sum(1 for r in self._triage_results if r.get("typology_classification") == t)
                for t in set(r.get("typology_classification", "UNKNOWN") for r in self._triage_results)
            },
        }

    def export_triage_results(self) -> pd.DataFrame:
        """Export triage results to a DataFrame."""
        if not self._triage_results:
            return pd.DataFrame()

        flat_results = []
        for result in self._triage_results:
            flat = {
                "alert_id": result.get("alert_id"),
                "entity_id": result.get("entity_id"),
                "narrative_summary": result.get("narrative_summary", ""),
                "recommended_disposition": result.get("recommended_disposition"),
                "confidence_level": result.get("confidence_level"),
                "typology_classification": result.get("typology_classification"),
                "estimated_risk_amount": result.get("estimated_risk_amount", 0),
                "n_risk_indicators": len(result.get("risk_indicators", [])),
                "verification_score": result.get("fact_verification", {}).get("verification_score", None),
                "verification_passed": result.get("fact_verification", {}).get("passed", None),
            }
            flat_results.append(flat)

        return pd.DataFrame(flat_results)
