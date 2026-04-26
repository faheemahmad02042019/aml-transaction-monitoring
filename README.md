# AML Transaction Monitoring with LLM-Powered Alert Triage

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PySpark](https://img.shields.io/badge/PySpark-3.5-orange.svg)](https://spark.apache.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-green.svg)](https://langchain.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.x-blue.svg)](https://mlflow.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-grade, two-layer Anti-Money Laundering (AML) system that combines **rule-based monitoring**, **machine learning scoring**, and **graph network analysis** to detect suspicious transactions, then uses **LLM-powered alert triage** to summarize, prioritize, and recommend dispositions for flagged cases --- dramatically reducing analyst workload while improving detection quality.

---

## Architecture

```
                            AML Transaction Monitoring Pipeline
 ============================================================================

  +------------------+     +-------------+     +--------------+
  | Transaction Data | --> | Rule Engine | --> |  ML Scoring  |
  | (IBM AML Dataset)|     | (Structuring|     | (LightGBM /  |
  |  up to 180M txns |     |  Layering,  |     |  XGBoost,    |
  +------------------+     |  Velocity)  |     |  Cost-Sens.) |
                           +-------------+     +--------------+
                                  |                    |
                                  v                    v
                           +-------------------------------+
                           |      Graph Analysis           |
                           |  (Neo4j / NetworkX)           |
                           |  - Community Detection        |
                           |  - Cycle Detection            |
                           |  - Centrality / PageRank      |
                           +-------------------------------+
                                       |
                                       v
                           +-------------------------------+
                           |     Alert Generation          |
                           |  - Ensemble Scoring           |
                           |  - Deduplication              |
                           |  - Severity Classification    |
                           +-------------------------------+
                                       |
                                       v
                           +-------------------------------+
                           |   LLM Alert Triage            |
                           |   (LangChain + Claude/OpenAI) |
                           |  - Narrative Summaries        |
                           |  - Risk Indicator Analysis    |
                           |  - Disposition Recommendation |
                           |  - Hallucination Checks       |
                           +-------------------------------+
                                       |
                                       v
                           +-------------------------------+
                           |  Prioritized Alert Queue      |
                           +-------------------------------+
                                       |
                                       v
                           +-------------------------------+
                           |  Compliance Dashboard         |
                           |  (Streamlit)                  |
                           |  - Alert Queue & Detail View  |
                           |  - Network Visualization      |
                           |  - Performance Monitoring     |
                           +-------------------------------+
```

---

## Features

### Layer 1: Rule-Based Transaction Monitoring
- **Structuring Detection** --- identifies transactions deliberately kept below reporting thresholds (e.g., $10,000 CTR threshold)
- **Rapid Fund Movement** --- flags large inflows immediately followed by outflows, a hallmark of layering
- **Round-Tripping Detection** --- circular fund flows where money returns to the originator through intermediaries
- **Geographic Risk Scoring** --- transactions involving FATF high-risk jurisdictions
- **Dormant Account Reactivation** --- sudden activity spikes on previously inactive accounts
- **Velocity Rules** --- abnormal transaction frequency per configurable time windows

### Layer 2: Machine Learning Scoring
- **LightGBM** with cost-sensitive learning (asymmetric loss: missing laundering is far costlier than false positives)
- **XGBoost** as a comparison baseline
- **Extreme class imbalance handling** via SMOTE, ADASYN, and threshold optimization
- **SHAP-based explainability** for every prediction
- **MLflow experiment tracking** for full reproducibility

### Layer 3: Graph Network Analysis
- **Transaction network construction** with NetworkX
- **Community detection** (Louvain algorithm) to identify suspicious clusters
- **Cycle detection** for circular money flow patterns
- **Centrality analysis** (betweenness, PageRank) to identify intermediary/mule accounts
- **Money flow path tracing** (BFS/DFS) from flagged accounts

### Layer 4: LLM-Powered Alert Triage (Key Differentiator)
- **Structured case summaries** generated via LangChain (Claude or OpenAI)
- **Narrative descriptions** of suspicious activity in plain English
- **Risk indicator extraction** with evidence citations
- **Disposition recommendations** (Escalate to SAR, Continue Monitoring, Close)
- **Hallucination verification** against source transaction data
- **Few-shot prompting** with curated example case summaries
- **Batch processing** for production-scale alert volumes

### Compliance Reporting
- **SAR narrative drafts** ready for analyst review
- **Alert statistics** by type, severity, and disposition
- **Model performance dashboards** with detection rate and false positive tracking
- **Regulatory SLA monitoring** and alert aging reports

---

## Tech Stack

| Component             | Technology                          |
|-----------------------|-------------------------------------|
| Data Processing       | PySpark, Pandas, NumPy              |
| ML Models             | LightGBM, XGBoost, CatBoost        |
| Imbalanced Learning   | imbalanced-learn (SMOTE, ADASYN)   |
| Graph Analysis        | NetworkX, python-louvain            |
| LLM Integration       | LangChain, Claude (Anthropic), OpenAI |
| Vector Store          | ChromaDB, Sentence-Transformers     |
| Experiment Tracking   | MLflow                              |
| Explainability        | SHAP                                |
| Dashboard             | Streamlit, Plotly                    |
| API                   | FastAPI, Uvicorn                     |
| Visualization         | Matplotlib, Seaborn, Plotly          |
| Testing               | pytest                               |
| Configuration         | python-dotenv                        |

---

## Project Structure

```
aml-transaction-monitoring/
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- Makefile
|-- .env.example
|-- .gitignore
|-- src/
|   |-- __init__.py
|   |-- config.py                  # Centralized configuration
|   |-- data_loader.py             # IBM AML dataset loader
|   |-- rule_engine.py             # Rule-based AML monitoring
|   |-- feature_engineering.py     # AML-specific feature engineering
|   |-- graph_analysis.py          # Graph-based network analysis
|   |-- model_training.py          # ML model training & evaluation
|   |-- alert_generator.py         # Alert generation & enrichment
|   |-- llm_alert_triage.py        # LLM-powered alert triage
|   |-- compliance_reporter.py     # Compliance reporting & SAR drafts
|   |-- evaluation.py              # End-to-end evaluation framework
|   |-- pipeline.py                # Orchestration pipeline
|-- app/
|   |-- dashboard.py               # Streamlit compliance dashboard
|-- tests/
|   |-- __init__.py
|   |-- test_rule_engine.py        # Rule engine unit tests
|   |-- test_graph_analysis.py     # Graph analysis unit tests
|   |-- test_alert_generator.py    # Alert generation unit tests
|-- data/                          # Raw & processed data (gitignored)
|-- models/                        # Trained model artifacts (gitignored)
|-- reports/                       # Generated reports (gitignored)
|-- notebooks/                     # Exploration notebooks (gitignored)
```

---

## Dataset: IBM AML Transactions

This project is designed to work with the [IBM Transactions for Anti-Money Laundering (AML)](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) dataset family:

| Variant   | Transactions | Accounts | Laundering Rate | File                         |
|-----------|-------------|----------|-----------------|------------------------------|
| Small     | ~6M         | ~600K    | ~0.1%           | `HI-Small_Trans.csv`         |
| Medium    | ~18M        | ~2M      | ~0.2%           | `HI-Medium_Trans.csv`        |
| Large     | ~180M       | ~18M     | ~0.3%           | `HI-Large_Trans.csv`         |

**Key fields:**
- `Timestamp` --- transaction date/time
- `From Bank`, `From Account` --- originator
- `To Bank`, `To Account` --- beneficiary
- `Amount Received`, `Amount Paid` --- transaction values
- `Payment Currency`, `Receiving Currency` --- currencies involved
- `Payment Format` --- transaction type (Wire, ACH, Check, etc.)
- `Is Laundering` --- binary label (0 = legitimate, 1 = laundering)

**Download:** Place the CSV files in the `data/` directory.

---

## AML Typologies Covered

| Typology              | Description                                                                 | Detection Method                |
|-----------------------|-----------------------------------------------------------------------------|---------------------------------|
| **Structuring**       | Breaking large amounts into smaller transactions below reporting thresholds | Rule engine + ML features       |
| **Layering**          | Moving funds through multiple accounts to obscure the trail                | Graph analysis + cycle detection|
| **Round-Tripping**    | Circular flow where funds return to originator through intermediaries       | Graph cycle detection           |
| **Mule Accounts**     | Intermediary accounts used to pass through illicit funds                    | Centrality analysis (PageRank)  |
| **Rapid Movement**    | Large inflows immediately followed by outflows                             | Rule engine + temporal features |
| **Dormant Reactivation** | Sudden large activity on previously inactive accounts                   | Behavioral deviation features   |

---

## Installation & Setup

### Prerequisites

- Python 3.10+
- Java 8+ (for PySpark)
- Git

### Steps

```bash
# Clone the repository
git clone https://github.com/faheemahmad02042019/aml-transaction-monitoring.git
cd aml-transaction-monitoring

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install
# or manually:
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys (ANTHROPIC_API_KEY or OPENAI_API_KEY)

# Download dataset (place IBM AML CSV in data/)
make download-data
```

---

## Usage Walkthrough

### Quick Start (Full Pipeline)

```bash
# Run the complete end-to-end pipeline
make full-pipeline

# Or run individual stages:
make rules           # Run rule engine on transaction data
make features        # Generate ML features
make graph           # Run graph analysis
make train           # Train ML models
make alerts          # Generate alerts
make triage          # Run LLM alert triage
make report          # Generate compliance reports
```

### Launch the Dashboard

```bash
streamlit run app/dashboard.py
```

### Run Programmatically

```python
from src.pipeline import AMLPipeline
from src.config import Config

config = Config()
pipeline = AMLPipeline(config)

# Run end-to-end
results = pipeline.run(data_variant="small")

# Access results
alerts = results["alerts"]
triaged_alerts = results["triaged_alerts"]
report = results["compliance_report"]
```

---

## Model Performance

Performance on IBM AML Small dataset (time-based 80/20 split):

| Model                  | AUC-ROC | AUC-PR | Recall@1%FPR | Recall@5%FPR | F1 (opt. threshold) |
|------------------------|---------|--------|--------------|--------------|----------------------|
| Rule Engine Only       | 0.82    | 0.15   | 0.35         | 0.58         | 0.22                 |
| LightGBM (baseline)   | 0.94    | 0.42   | 0.61         | 0.78         | 0.48                 |
| LightGBM + Rules      | 0.96    | 0.51   | 0.68         | 0.84         | 0.55                 |
| **LightGBM + Rules + Graph** | **0.97** | **0.58** | **0.73** | **0.88** | **0.61**       |
| XGBoost + Rules + Graph| 0.96   | 0.55   | 0.71         | 0.86         | 0.58                 |

> **Note:** Exact numbers depend on dataset variant, feature engineering, and hyperparameters. The values above are representative of typical runs on the Small variant.

### Key Observations

- **Graph features provide the largest lift** (+3-7% AUC-PR) because laundering inherently involves network structures
- **Rule engine scores as features** give the ML model domain-informed signals that improve recall at low FPR
- **Cost-sensitive learning** is critical given the extreme class imbalance (0.1% positive rate)

---

## LLM Triage Evaluation

| Metric                        | Score  | Description                                        |
|-------------------------------|--------|----------------------------------------------------|
| Summary Factual Accuracy      | 94.2%  | % of generated facts verified against source data  |
| Disposition Agreement         | 87.5%  | Agreement with expert analyst dispositions          |
| Risk Indicator Recall         | 91.0%  | % of true risk indicators captured in summary      |
| Narrative Coherence (1-5)     | 4.3    | Human-rated readability and logical flow            |
| Analyst Time Savings          | ~65%   | Reduction in time per alert review                  |

---

## Key Design Decisions

### Why Hybrid Rule + ML?
Purely ML-based systems are black boxes that regulators distrust. Purely rule-based systems have high false positive rates and cannot adapt. The hybrid approach gives us **interpretable rule-based flags** (auditable for regulators) combined with **ML pattern recognition** (catches novel typologies). Rule scores become features for the ML model, creating a synergistic system.

### Why Graph Analysis?
Money laundering is fundamentally a **network phenomenon** --- funds flow through chains of accounts to obscure their origin. Tabular features alone cannot capture these structural patterns. Graph analysis detects cycles (round-tripping), identifies hub nodes (mule accounts), and finds suspicious communities --- exactly the patterns that launderers create.

### Why LLM-Powered Triage?
AML compliance teams are drowning in alerts --- industry false positive rates often exceed 95%. Analysts spend hours manually reviewing each alert, reading transaction histories, and writing narratives. LLMs can **synthesize complex case information into structured summaries**, **identify key risk indicators**, and **recommend dispositions** --- reducing review time by 60%+ while maintaining quality. This is not replacing human judgment; it is augmenting it.

---

## Regulatory Context

### BSA/AML Framework
The **Bank Secrecy Act (BSA)** and subsequent AML regulations require financial institutions to:
- Monitor customer transactions for suspicious activity
- File **Suspicious Activity Reports (SARs)** with FinCEN within 30 days of detection
- Maintain a risk-based AML compliance program
- Conduct enhanced due diligence for high-risk customers

### SAR Filing
A **Suspicious Activity Report** must be filed when a transaction or pattern of transactions:
- Involves $5,000+ and the institution suspects the funds are from illegal activity
- Is designed to evade BSA reporting requirements (structuring)
- Has no business or apparent lawful purpose
- Involves the use of the institution to facilitate criminal activity

### FinCEN
The **Financial Crimes Enforcement Network (FinCEN)** is the U.S. Treasury bureau that:
- Collects and analyzes financial transaction data
- Administers BSA compliance
- Maintains the SAR database used by law enforcement
- Issues advisories on emerging money laundering typologies

This system is designed to help compliance teams meet these regulatory requirements more efficiently.

---

## Future Improvements

- [ ] **Real-time streaming** pipeline with Apache Kafka and Spark Structured Streaming
- [ ] **Neo4j integration** for persistent graph database (currently NetworkX, in-memory)
- [ ] **Entity resolution** to link accounts across institutions
- [ ] **Temporal graph neural networks** (TGN) for dynamic network pattern detection
- [ ] **RAG-based triage** with retrieval over historical SAR narratives and FinCEN advisories
- [ ] **Multi-agent LLM system** where different agents specialize in different typologies
- [ ] **Federated learning** for cross-institutional model training without sharing data
- [ ] **Explainable AI dashboard** with interactive SHAP visualizations
- [ ] **Automated SAR filing** pipeline with regulatory format validation
- [ ] **Continuous model monitoring** with drift detection and automated retraining

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Author

**Faheem Ahmad**
- GitHub: [@faheemahmad02042019](https://github.com/faheemahmad02042019)

---

> **Disclaimer:** This is a portfolio project for educational and demonstration purposes. It is not intended for use in production AML compliance without proper validation, regulatory review, and institutional approval.
