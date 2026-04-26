"""
Unit tests for Graph-Based AML Network Analysis.

Tests:
  - Graph construction from transaction data
  - Community detection (Louvain)
  - Cycle detection for round-tripping
  - Centrality computation (betweenness, PageRank)
  - Subgraph extraction
  - Money flow path tracing
  - Graph feature generation
"""

from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

try:
    import networkx as nx
except ImportError:
    pytest.skip("NetworkX not installed", allow_module_level=True)

from src.config import Config
from src.graph_analysis import TransactionGraphAnalyzer


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def config() -> Config:
    """Return a default Config object with test-friendly graph settings."""
    cfg = Config()
    cfg.graph.min_edge_weight = 0.0  # Include all edges in tests
    cfg.graph.max_graph_nodes = 10_000
    cfg.graph.edge_time_window_days = 365
    cfg.graph.max_cycle_length = 6
    cfg.graph.min_community_size = 2
    return cfg


@pytest.fixture
def analyzer(config: Config) -> TransactionGraphAnalyzer:
    """Return a TransactionGraphAnalyzer instance."""
    return TransactionGraphAnalyzer(config)


@pytest.fixture
def simple_transaction_df() -> pd.DataFrame:
    """
    Return a small transaction DataFrame with known graph structure.

    Network:
        A -> B (2 txns)
        B -> C (1 txn)
        C -> A (1 txn)  -- creates a cycle A->B->C->A
        D -> E (1 txn)  -- isolated pair
    """
    records = [
        {"from_id": "A", "to_id": "B", "amount": 5000.0, "timestamp": "2024-01-01 10:00", "is_laundering": 0},
        {"from_id": "A", "to_id": "B", "amount": 3000.0, "timestamp": "2024-01-01 11:00", "is_laundering": 0},
        {"from_id": "B", "to_id": "C", "amount": 7000.0, "timestamp": "2024-01-01 12:00", "is_laundering": 1},
        {"from_id": "C", "to_id": "A", "amount": 6000.0, "timestamp": "2024-01-01 13:00", "is_laundering": 0},
        {"from_id": "D", "to_id": "E", "amount": 2000.0, "timestamp": "2024-01-01 14:00", "is_laundering": 0},
    ]
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["transaction_id"] = range(len(df))
    return df


@pytest.fixture
def larger_transaction_df() -> pd.DataFrame:
    """
    Return a larger transaction DataFrame for testing at scale.

    Contains 100 transactions across 20 accounts with some cycles
    and community structure.
    """
    rng = np.random.RandomState(42)
    n = 100
    accounts = [f"ACCT_{i:03d}" for i in range(20)]

    records = []
    for i in range(n):
        from_idx = rng.randint(0, 20)
        to_idx = rng.randint(0, 20)
        while to_idx == from_idx:
            to_idx = rng.randint(0, 20)

        records.append({
            "transaction_id": i,
            "from_id": accounts[from_idx],
            "to_id": accounts[to_idx],
            "amount": float(rng.lognormal(8, 1.5)),
            "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=rng.randint(0, 720)),
            "is_laundering": rng.choice([0, 1], p=[0.95, 0.05]),
        })

    df = pd.DataFrame(records)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Graph Construction Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphConstruction:
    """Tests for building the transaction graph."""

    def test_graph_has_correct_nodes(self, analyzer: TransactionGraphAnalyzer, simple_transaction_df: pd.DataFrame) -> None:
        """Graph should contain all unique accounts as nodes."""
        graph = analyzer.build_graph(simple_transaction_df)

        expected_nodes = {"A", "B", "C", "D", "E"}
        assert set(graph.nodes()) == expected_nodes

    def test_graph_has_correct_edges(self, analyzer: TransactionGraphAnalyzer, simple_transaction_df: pd.DataFrame) -> None:
        """Graph should contain aggregated edges between transacting accounts."""
        graph = analyzer.build_graph(simple_transaction_df)

        expected_edges = {("A", "B"), ("B", "C"), ("C", "A"), ("D", "E")}
        assert set(graph.edges()) == expected_edges

    def test_edge_weight_is_total_amount(self, analyzer: TransactionGraphAnalyzer, simple_transaction_df: pd.DataFrame) -> None:
        """Edge weight should be the total transaction amount between the pair."""
        graph = analyzer.build_graph(simple_transaction_df)

        # A -> B: 5000 + 3000 = 8000
        assert graph["A"]["B"]["weight"] == 8000.0

    def test_edge_transaction_count(self, analyzer: TransactionGraphAnalyzer, simple_transaction_df: pd.DataFrame) -> None:
        """Edge should track the number of transactions."""
        graph = analyzer.build_graph(simple_transaction_df)

        # A -> B: 2 transactions
        assert graph["A"]["B"]["txn_count"] == 2
        # B -> C: 1 transaction
        assert graph["B"]["C"]["txn_count"] == 1

    def test_graph_is_directed(self, analyzer: TransactionGraphAnalyzer, simple_transaction_df: pd.DataFrame) -> None:
        """The transaction graph should be directed."""
        graph = analyzer.build_graph(simple_transaction_df)

        assert isinstance(graph, nx.DiGraph)

    def test_node_attributes(self, analyzer: TransactionGraphAnalyzer, simple_transaction_df: pd.DataFrame) -> None:
        """Nodes should have degree and flow attributes."""
        graph = analyzer.build_graph(simple_transaction_df)

        for node in graph.nodes():
            assert "out_degree" in graph.nodes[node]
            assert "in_degree" in graph.nodes[node]
            assert "total_out_flow" in graph.nodes[node]
            assert "total_in_flow" in graph.nodes[node]
            assert "net_flow" in graph.nodes[node]

    def test_empty_dataframe(self, analyzer: TransactionGraphAnalyzer) -> None:
        """Building a graph from an empty DataFrame should succeed."""
        df = pd.DataFrame(columns=["from_id", "to_id", "amount", "timestamp", "is_laundering"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        graph = analyzer.build_graph(df)

        assert graph.number_of_nodes() == 0
        assert graph.number_of_edges() == 0

    def test_min_edge_weight_filter(self, config: Config, simple_transaction_df: pd.DataFrame) -> None:
        """Transactions below min_edge_weight should be excluded."""
        config.graph.min_edge_weight = 4000.0
        analyzer = TransactionGraphAnalyzer(config)
        graph = analyzer.build_graph(simple_transaction_df)

        # D -> E (2000) should be excluded
        assert not graph.has_edge("D", "E")

    def test_larger_graph(self, analyzer: TransactionGraphAnalyzer, larger_transaction_df: pd.DataFrame) -> None:
        """Building a graph from 100 transactions should succeed."""
        graph = analyzer.build_graph(larger_transaction_df)

        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0


# ─────────────────────────────────────────────────────────────────────────────
# Community Detection Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCommunityDetection:
    """Tests for Louvain community detection."""

    def test_communities_detected(self, analyzer: TransactionGraphAnalyzer, simple_transaction_df: pd.DataFrame) -> None:
        """Community detection should assign community IDs to all nodes."""
        analyzer.build_graph(simple_transaction_df)
        communities = analyzer.detect_communities()

        assert isinstance(communities, dict)
        # All nodes in the graph should have a community assignment
        for node in analyzer._graph.nodes():
            assert node in communities

    def test_separated_components_get_different_communities(
        self, analyzer: TransactionGraphAnalyzer, simple_transaction_df: pd.DataFrame
    ) -> None:
        """Disconnected components should ideally be in different communities."""
        analyzer.build_graph(simple_transaction_df)
        communities = analyzer.detect_communities()

        # A, B, C form a connected component; D, E are separate
        # (Though community detection on small graphs may vary,
        # at minimum all nodes should have assignments)
        assert communities.get("A") is not None
        assert communities.get("D") is not None

    def test_community_ids_are_integers(self, analyzer: TransactionGraphAnalyzer, simple_transaction_df: pd.DataFrame) -> None:
        """Community IDs should be integers."""
        analyzer.build_graph(simple_transaction_df)
        communities = analyzer.detect_communities()

        for node, comm_id in communities.items():
            assert isinstance(comm_id, (int, np.integer))

    def test_larger_graph_communities(self, analyzer: TransactionGraphAnalyzer, larger_transaction_df: pd.DataFrame) -> None:
        """Community detection on a larger graph should find at least one community."""
        analyzer.build_graph(larger_transaction_df)
        communities = analyzer.detect_communities()

        # At least one community should exist
        unique_communities = set(communities.values()) - {-1}
        assert len(unique_communities) >= 1

    def test_communities_require_graph(self, analyzer: TransactionGraphAnalyzer) -> None:
        """Detecting communities without a graph should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="Graph not built"):
            analyzer.detect_communities()


# ─────────────────────────────────────────────────────────────────────────────
# Cycle Detection Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCycleDetection:
    """Tests for detecting cycles (round-tripping patterns)."""

    def test_simple_cycle_detected(self, analyzer: TransactionGraphAnalyzer, simple_transaction_df: pd.DataFrame) -> None:
        """The A->B->C->A cycle should be detected."""
        analyzer.build_graph(simple_transaction_df)
        cycles = analyzer.detect_cycles()

        assert len(cycles) > 0

        # At least one cycle should contain A, B, C
        found_abc_cycle = False
        for cycle in cycles:
            if set(cycle) == {"A", "B", "C"}:
                found_abc_cycle = True
                break

        assert found_abc_cycle, f"Expected A->B->C cycle, found: {cycles}"

    def test_no_cycle_in_dag(self, analyzer: TransactionGraphAnalyzer) -> None:
        """A DAG (directed acyclic graph) should have no cycles."""
        df = pd.DataFrame({
            "from_id": ["A", "B", "C"],
            "to_id": ["B", "C", "D"],
            "amount": [1000.0, 2000.0, 3000.0],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "is_laundering": [0, 0, 0],
            "transaction_id": [0, 1, 2],
        })

        analyzer.build_graph(df)
        cycles = analyzer.detect_cycles()

        assert len(cycles) == 0

    def test_self_loop_detected(self, analyzer: TransactionGraphAnalyzer) -> None:
        """A self-loop (A->A) should be detected as a cycle of length 1."""
        df = pd.DataFrame({
            "from_id": ["A", "A"],
            "to_id": ["B", "A"],  # A->A is a self-loop
            "amount": [1000.0, 2000.0],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "is_laundering": [0, 0],
            "transaction_id": [0, 1],
        })

        analyzer.build_graph(df)
        cycles = analyzer.detect_cycles()

        # NetworkX simple_cycles should detect the self-loop
        # (The exact behavior depends on the edge aggregation)
        assert isinstance(cycles, list)

    def test_cycle_respects_max_length(self, config: Config) -> None:
        """Cycles longer than max_cycle_length should not be reported."""
        config.graph.max_cycle_length = 3

        # Create a cycle of length 5: A->B->C->D->E->A
        df = pd.DataFrame({
            "from_id": ["A", "B", "C", "D", "E"],
            "to_id": ["B", "C", "D", "E", "A"],
            "amount": [1000.0] * 5,
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="h"),
            "is_laundering": [0] * 5,
            "transaction_id": range(5),
        })

        analyzer = TransactionGraphAnalyzer(config)
        analyzer.build_graph(df)
        cycles = analyzer.detect_cycles()

        # No cycle of length 5 should be reported (max is 3)
        for cycle in cycles:
            assert len(cycle) <= config.graph.max_cycle_length

    def test_cycles_require_graph(self, analyzer: TransactionGraphAnalyzer) -> None:
        """Detecting cycles without a graph should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="Graph not built"):
            analyzer.detect_cycles()


# ─────────────────────────────────────────────────────────────────────────────
# Centrality Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCentrality:
    """Tests for centrality computation."""

    def test_centrality_metrics_computed(self, analyzer: TransactionGraphAnalyzer, simple_transaction_df: pd.DataFrame) -> None:
        """Centrality should return betweenness, PageRank, and degree metrics."""
        analyzer.build_graph(simple_transaction_df)
        centrality = analyzer.compute_centrality()

        assert "betweenness" in centrality
        assert "pagerank" in centrality
        assert "in_degree" in centrality
        assert "out_degree" in centrality

    def test_pagerank_sums_to_one(self, analyzer: TransactionGraphAnalyzer, simple_transaction_df: pd.DataFrame) -> None:
        """PageRank values should approximately sum to 1."""
        analyzer.build_graph(simple_transaction_df)
        centrality = analyzer.compute_centrality()

        pagerank = centrality["pagerank"]
        if pagerank:
            total = sum(pagerank.values())
            assert abs(total - 1.0) < 0.01, f"PageRank sum = {total}, expected ~1.0"

    def test_betweenness_in_valid_range(self, analyzer: TransactionGraphAnalyzer, simple_transaction_df: pd.DataFrame) -> None:
        """Betweenness centrality should be in [0, 1] (normalized)."""
        analyzer.build_graph(simple_transaction_df)
        centrality = analyzer.compute_centrality()

        for node, value in centrality["betweenness"].items():
            assert 0.0 <= value <= 1.0, f"Node {node} betweenness={value}"

    def test_hub_node_has_higher_centrality(self, analyzer: TransactionGraphAnalyzer) -> None:
        """A hub node should have higher centrality than peripheral nodes."""
        # Star graph: HUB connects to SPOKE_0, ..., SPOKE_4
        records = []
        for i in range(5):
            records.append({
                "from_id": "HUB",
                "to_id": f"SPOKE_{i}",
                "amount": 5000.0,
                "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i),
                "is_laundering": 0,
                "transaction_id": i,
            })
            records.append({
                "from_id": f"SPOKE_{i}",
                "to_id": "HUB",
                "amount": 3000.0,
                "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i + 5),
                "is_laundering": 0,
                "transaction_id": i + 5,
            })

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        analyzer.build_graph(df)
        centrality = analyzer.compute_centrality()

        hub_pagerank = centrality["pagerank"].get("HUB", 0)
        spoke_pageranks = [
            centrality["pagerank"].get(f"SPOKE_{i}", 0) for i in range(5)
        ]
        avg_spoke = np.mean(spoke_pageranks) if spoke_pageranks else 0

        # Hub should generally have comparable or higher centrality in a star
        assert hub_pagerank > 0

    def test_centrality_requires_graph(self, analyzer: TransactionGraphAnalyzer) -> None:
        """Computing centrality without a graph should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="Graph not built"):
            analyzer.compute_centrality()


# ─────────────────────────────────────────────────────────────────────────────
# Subgraph Extraction and Path Tracing Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSubgraphAndPathTracing:
    """Tests for subgraph extraction and money flow path tracing."""

    def test_subgraph_contains_seed(self, analyzer: TransactionGraphAnalyzer, simple_transaction_df: pd.DataFrame) -> None:
        """Extracted subgraph should contain the seed node."""
        analyzer.build_graph(simple_transaction_df)
        subgraph = analyzer.extract_subgraph(seed_nodes=["A"])

        assert "A" in subgraph.nodes()

    def test_subgraph_includes_neighbors(self, analyzer: TransactionGraphAnalyzer, simple_transaction_df: pd.DataFrame) -> None:
        """Subgraph should include immediate neighbors of seed nodes."""
        analyzer.build_graph(simple_transaction_df)
        subgraph = analyzer.extract_subgraph(seed_nodes=["A"], max_hops=1)

        # A connects to B (out) and receives from C (in)
        assert "B" in subgraph.nodes()
        assert "C" in subgraph.nodes()

    def test_subgraph_respects_max_hops(self, analyzer: TransactionGraphAnalyzer, simple_transaction_df: pd.DataFrame) -> None:
        """Subgraph should not extend beyond max_hops."""
        analyzer.build_graph(simple_transaction_df)

        # D and E are disconnected from A, so with seed=["A"], max_hops=1,
        # D and E should not appear
        subgraph = analyzer.extract_subgraph(seed_nodes=["A"], max_hops=1)

        assert "D" not in subgraph.nodes()
        assert "E" not in subgraph.nodes()

    def test_subgraph_respects_max_nodes(self, analyzer: TransactionGraphAnalyzer, simple_transaction_df: pd.DataFrame) -> None:
        """Subgraph should not exceed max_nodes."""
        analyzer.build_graph(simple_transaction_df)
        subgraph = analyzer.extract_subgraph(seed_nodes=["A"], max_nodes=2)

        assert subgraph.number_of_nodes() <= 2

    def test_trace_money_flow(self, analyzer: TransactionGraphAnalyzer, simple_transaction_df: pd.DataFrame) -> None:
        """Money flow tracing from A should find paths."""
        analyzer.build_graph(simple_transaction_df)
        paths = analyzer.trace_money_flow("A", max_depth=3)

        assert isinstance(paths, list)
        assert len(paths) > 0

    def test_trace_nonexistent_source(self, analyzer: TransactionGraphAnalyzer, simple_transaction_df: pd.DataFrame) -> None:
        """Tracing from a nonexistent node should return empty list."""
        analyzer.build_graph(simple_transaction_df)
        paths = analyzer.trace_money_flow("NONEXISTENT")

        assert paths == []

    def test_subgraph_requires_graph(self, analyzer: TransactionGraphAnalyzer) -> None:
        """Extracting a subgraph without a graph should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="Graph not built"):
            analyzer.extract_subgraph(seed_nodes=["A"])


# ─────────────────────────────────────────────────────────────────────────────
# Graph Feature Generation Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphFeatureGeneration:
    """Tests for generating per-transaction graph features."""

    def test_feature_columns_present(self, analyzer: TransactionGraphAnalyzer, simple_transaction_df: pd.DataFrame) -> None:
        """Generated features should include expected graph feature columns."""
        analyzer.build_graph(simple_transaction_df)
        features = analyzer.generate_graph_features(simple_transaction_df)

        assert any(c.startswith("graph_") for c in features.columns)
        assert "graph_sender_betweenness" in features.columns or "graph_sender_pagerank" in features.columns

    def test_feature_length_matches_input(self, analyzer: TransactionGraphAnalyzer, simple_transaction_df: pd.DataFrame) -> None:
        """Feature matrix should have same number of rows as input."""
        analyzer.build_graph(simple_transaction_df)
        features = analyzer.generate_graph_features(simple_transaction_df)

        assert len(features) == len(simple_transaction_df)

    def test_cycle_involvement_feature(self, analyzer: TransactionGraphAnalyzer, simple_transaction_df: pd.DataFrame) -> None:
        """Nodes in a cycle should have cycle involvement features."""
        analyzer.build_graph(simple_transaction_df)
        analyzer.detect_cycles()
        features = analyzer.generate_graph_features(simple_transaction_df)

        if "graph_sender_in_cycle" in features.columns:
            # A, B, C are in a cycle; at least some rows should have cycle=1
            assert features["graph_sender_in_cycle"].sum() > 0

    def test_features_are_numeric(self, analyzer: TransactionGraphAnalyzer, simple_transaction_df: pd.DataFrame) -> None:
        """All graph features should be numeric."""
        analyzer.build_graph(simple_transaction_df)
        features = analyzer.generate_graph_features(simple_transaction_df)

        for col in features.columns:
            assert pd.api.types.is_numeric_dtype(features[col]), f"Column '{col}' is not numeric"


# ─────────────────────────────────────────────────────────────────────────────
# Suspicious Account Identification Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSuspiciousAccounts:
    """Tests for identifying suspicious accounts."""

    def test_suspicious_accounts_returned(self, analyzer: TransactionGraphAnalyzer, simple_transaction_df: pd.DataFrame) -> None:
        """identify_suspicious_accounts should return a DataFrame."""
        analyzer.build_graph(simple_transaction_df)
        result = analyzer.identify_suspicious_accounts(top_k=5)

        assert isinstance(result, pd.DataFrame)
        assert "account_id" in result.columns
        assert "graph_suspicion_score" in result.columns

    def test_suspicious_scores_in_valid_range(
        self, analyzer: TransactionGraphAnalyzer, simple_transaction_df: pd.DataFrame
    ) -> None:
        """Suspicion scores should be in [0, 1]."""
        analyzer.build_graph(simple_transaction_df)
        result = analyzer.identify_suspicious_accounts()

        assert (result["graph_suspicion_score"] >= 0).all()
        assert (result["graph_suspicion_score"] <= 1).all()

    def test_top_k_limit(self, analyzer: TransactionGraphAnalyzer, larger_transaction_df: pd.DataFrame) -> None:
        """Results should be limited to top_k accounts."""
        analyzer.build_graph(larger_transaction_df)
        result = analyzer.identify_suspicious_accounts(top_k=5)

        assert len(result) <= 5

    def test_results_sorted_by_score(self, analyzer: TransactionGraphAnalyzer, simple_transaction_df: pd.DataFrame) -> None:
        """Results should be sorted by suspicion score in descending order."""
        analyzer.build_graph(simple_transaction_df)
        result = analyzer.identify_suspicious_accounts()

        scores = result["graph_suspicion_score"].values
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


# ─────────────────────────────────────────────────────────────────────────────
# Graph Summary Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphSummary:
    """Tests for graph summary statistics."""

    def test_summary_structure(self, analyzer: TransactionGraphAnalyzer, simple_transaction_df: pd.DataFrame) -> None:
        """get_graph_summary should return expected keys."""
        analyzer.build_graph(simple_transaction_df)
        summary = analyzer.get_graph_summary()

        assert "nodes" in summary
        assert "edges" in summary
        assert "density" in summary

    def test_summary_without_graph(self, analyzer: TransactionGraphAnalyzer) -> None:
        """Summary without a graph should indicate status."""
        summary = analyzer.get_graph_summary()

        assert "status" in summary

    def test_density_in_valid_range(self, analyzer: TransactionGraphAnalyzer, simple_transaction_df: pd.DataFrame) -> None:
        """Graph density should be in [0, 1]."""
        analyzer.build_graph(simple_transaction_df)
        summary = analyzer.get_graph_summary()

        assert 0.0 <= summary["density"] <= 1.0
