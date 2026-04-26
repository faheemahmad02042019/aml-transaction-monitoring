"""
Graph-Based AML Network Analysis.

Constructs transaction networks and applies graph algorithms to detect
money laundering patterns that are invisible in tabular data:

  - Community detection (Louvain) to find suspicious clusters
  - Cycle detection for round-tripping / circular fund flows
  - Centrality analysis (betweenness, PageRank) for mule account identification
  - Subgraph extraction for focused investigation
  - Money flow path tracing (BFS/DFS) from flagged accounts
"""

import logging
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

try:
    import networkx as nx
except ImportError:
    raise ImportError("NetworkX is required: pip install networkx")

try:
    import community as community_louvain
except ImportError:
    community_louvain = None

from src.config import Config, GraphAnalysisConfig

logger = logging.getLogger(__name__)


class TransactionGraphAnalyzer:
    """
    Graph-based analysis of transaction networks for AML detection.

    Builds a directed, weighted graph from transaction data and applies
    graph algorithms to identify suspicious network structures.

    Parameters
    ----------
    config : Config
        Master configuration object.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.graph_config: GraphAnalysisConfig = config.graph
        self._graph: Optional[nx.DiGraph] = None
        self._communities: Optional[Dict[str, int]] = None
        self._centrality: Optional[Dict[str, Dict[str, float]]] = None
        self._cycles: List[List[str]] = []

    def build_graph(self, df: pd.DataFrame) -> nx.DiGraph:
        """
        Build a directed, weighted transaction network graph.

        Nodes are accounts, edges represent aggregated fund flows. Edge
        attributes include total amount, transaction count, and date range.

        Parameters
        ----------
        df : pd.DataFrame
            Transaction data with from_id, to_id, amount, timestamp.

        Returns
        -------
        nx.DiGraph
            Directed transaction graph.
        """
        logger.info(f"Building transaction graph from {len(df):,} transactions")
        cfg = self.graph_config

        # Filter by minimum edge weight and time window
        filtered = df[df["amount"] >= cfg.min_edge_weight].copy()
        if cfg.edge_time_window_days > 0:
            cutoff = filtered["timestamp"].max() - pd.Timedelta(days=cfg.edge_time_window_days)
            filtered = filtered[filtered["timestamp"] >= cutoff]

        logger.info(f"  Filtered to {len(filtered):,} transactions (min_amount={cfg.min_edge_weight}, window={cfg.edge_time_window_days}d)")

        # Aggregate edges: (from_id, to_id) -> total_amount, count, dates
        edge_agg = (
            filtered.groupby(["from_id", "to_id"])
            .agg(
                total_amount=("amount", "sum"),
                txn_count=("amount", "count"),
                mean_amount=("amount", "mean"),
                max_amount=("amount", "max"),
                first_txn=("timestamp", "min"),
                last_txn=("timestamp", "max"),
                any_laundering=("is_laundering", "max"),
            )
            .reset_index()
        )

        # Build the graph
        G = nx.DiGraph()

        # Limit graph size for memory safety
        if len(edge_agg) > cfg.max_graph_nodes * 10:
            logger.warning(
                f"Edge count ({len(edge_agg):,}) exceeds limit. "
                f"Sampling top edges by total_amount."
            )
            edge_agg = edge_agg.nlargest(cfg.max_graph_nodes * 5, "total_amount")

        for _, row in edge_agg.iterrows():
            G.add_edge(
                row["from_id"],
                row["to_id"],
                weight=row["total_amount"],
                txn_count=int(row["txn_count"]),
                mean_amount=row["mean_amount"],
                max_amount=row["max_amount"],
                first_txn=str(row["first_txn"]),
                last_txn=str(row["last_txn"]),
                any_laundering=int(row["any_laundering"]),
            )

        # Compute node-level attributes
        for node in G.nodes():
            G.nodes[node]["out_degree"] = G.out_degree(node)
            G.nodes[node]["in_degree"] = G.in_degree(node)
            G.nodes[node]["total_out_flow"] = sum(
                G[node][succ]["weight"] for succ in G.successors(node)
            )
            G.nodes[node]["total_in_flow"] = sum(
                G[pred][node]["weight"] for pred in G.predecessors(node)
            )
            G.nodes[node]["net_flow"] = (
                G.nodes[node]["total_in_flow"] - G.nodes[node]["total_out_flow"]
            )

        self._graph = G
        logger.info(
            f"  Graph built: {G.number_of_nodes():,} nodes, "
            f"{G.number_of_edges():,} edges"
        )

        return G

    def detect_communities(self) -> Dict[str, int]:
        """
        Detect communities in the transaction network using the Louvain algorithm.

        Communities may indicate groups of accounts involved in coordinated
        suspicious activity (e.g., money laundering networks).

        Returns
        -------
        Dict[str, int]
            Mapping of node (account) to community ID.
        """
        if self._graph is None:
            raise RuntimeError("Graph not built. Call build_graph() first.")

        if community_louvain is None:
            logger.warning("python-louvain not installed. Skipping community detection.")
            self._communities = {node: 0 for node in self._graph.nodes()}
            return self._communities

        logger.info("Running Louvain community detection")

        # Convert to undirected for Louvain (it requires undirected graph)
        G_undirected = self._graph.to_undirected()

        partition = community_louvain.best_partition(
            G_undirected,
            resolution=self.graph_config.louvain_resolution,
            random_state=42,
        )

        # Filter to significant communities
        community_sizes = defaultdict(int)
        for node, comm_id in partition.items():
            community_sizes[comm_id] += 1

        significant_communities = {
            comm_id
            for comm_id, size in community_sizes.items()
            if size >= self.graph_config.min_community_size
        }

        # Assign -1 to nodes in insignificant communities
        cleaned_partition = {
            node: (comm_id if comm_id in significant_communities else -1)
            for node, comm_id in partition.items()
        }

        self._communities = cleaned_partition

        n_communities = len(significant_communities)
        modularity = community_louvain.modularity(partition, G_undirected)

        logger.info(
            f"  Communities detected: {n_communities} significant "
            f"(min_size={self.graph_config.min_community_size}) | "
            f"Modularity: {modularity:.4f}"
        )

        # Set community IDs as node attributes
        nx.set_node_attributes(self._graph, cleaned_partition, "community")

        return cleaned_partition

    def detect_cycles(self) -> List[List[str]]:
        """
        Detect cycles (circular fund flows) in the transaction graph.

        Cycles of length 2-N indicate potential round-tripping, where
        money flows in a circle back to the originator.

        Returns
        -------
        List[List[str]]
            List of detected cycles (each cycle is a list of account IDs).
        """
        if self._graph is None:
            raise RuntimeError("Graph not built. Call build_graph() first.")

        logger.info("Detecting cycles in transaction graph")
        cfg = self.graph_config
        all_cycles: List[List[str]] = []

        try:
            # Use Johnson's algorithm for finding all simple cycles
            # Limit by max cycle length
            cycle_generator = nx.simple_cycles(self._graph)

            count = 0
            for cycle in cycle_generator:
                if len(cycle) <= cfg.max_cycle_length and len(cycle) >= 2:
                    all_cycles.append(cycle)
                    count += 1

                # Safety limit to prevent infinite computation
                if count >= 10_000:
                    logger.warning("Cycle detection limit reached (10,000 cycles)")
                    break

        except Exception as e:
            logger.warning(f"Cycle detection encountered an error: {e}")

        self._cycles = all_cycles

        # Categorize cycles by length
        cycle_lengths = defaultdict(int)
        for cycle in all_cycles:
            cycle_lengths[len(cycle)] += 1

        logger.info(
            f"  Detected {len(all_cycles)} cycles | "
            f"Length distribution: {dict(cycle_lengths)}"
        )

        return all_cycles

    def compute_centrality(self) -> Dict[str, Dict[str, float]]:
        """
        Compute centrality metrics to identify key intermediary/mule accounts.

        Metrics:
          - Betweenness centrality: identifies nodes that bridge communities
          - PageRank: identifies "important" nodes in the flow network
          - In/Out degree centrality: simple connectivity measures

        Returns
        -------
        Dict[str, Dict[str, float]]
            Nested dict: metric_name -> {node: value}.
        """
        if self._graph is None:
            raise RuntimeError("Graph not built. Call build_graph() first.")

        logger.info("Computing centrality metrics")
        centrality = {}

        # Betweenness centrality
        logger.info("  Computing betweenness centrality")
        try:
            betweenness = nx.betweenness_centrality(
                self._graph,
                weight="weight",
                normalized=True,
            )
            centrality["betweenness"] = betweenness
        except Exception as e:
            logger.warning(f"Betweenness centrality failed: {e}")
            centrality["betweenness"] = {}

        # PageRank
        logger.info("  Computing PageRank")
        try:
            pagerank = nx.pagerank(
                self._graph,
                alpha=self.graph_config.pagerank_alpha,
                max_iter=self.graph_config.pagerank_max_iter,
                weight="weight",
            )
            centrality["pagerank"] = pagerank
        except Exception as e:
            logger.warning(f"PageRank failed: {e}")
            centrality["pagerank"] = {}

        # In-degree centrality
        in_degree = nx.in_degree_centrality(self._graph)
        centrality["in_degree"] = in_degree

        # Out-degree centrality
        out_degree = nx.out_degree_centrality(self._graph)
        centrality["out_degree"] = out_degree

        self._centrality = centrality

        # Log top nodes
        for metric_name, values in centrality.items():
            if values:
                top_nodes = sorted(values.items(), key=lambda x: x[1], reverse=True)[:5]
                logger.info(f"  Top-5 {metric_name}: {[(n, round(v, 4)) for n, v in top_nodes]}")

        return centrality

    def extract_subgraph(
        self,
        seed_nodes: List[str],
        max_hops: Optional[int] = None,
        max_nodes: Optional[int] = None,
    ) -> nx.DiGraph:
        """
        Extract a subgraph around seed nodes for focused investigation.

        Parameters
        ----------
        seed_nodes : List[str]
            Starting account IDs for subgraph extraction.
        max_hops : int, optional
            Maximum number of hops from seed nodes. Defaults to config value.
        max_nodes : int, optional
            Maximum subgraph size. Defaults to config value.

        Returns
        -------
        nx.DiGraph
            Extracted subgraph.
        """
        if self._graph is None:
            raise RuntimeError("Graph not built. Call build_graph() first.")

        max_hops = max_hops or self.graph_config.subgraph_max_hops
        max_nodes = max_nodes or self.graph_config.subgraph_max_nodes

        logger.info(f"Extracting subgraph: {len(seed_nodes)} seed nodes, max_hops={max_hops}")

        # BFS from seed nodes
        visited: Set[str] = set()
        queue: deque = deque()

        for node in seed_nodes:
            if node in self._graph:
                queue.append((node, 0))
                visited.add(node)

        while queue and len(visited) < max_nodes:
            current, depth = queue.popleft()

            if depth >= max_hops:
                continue

            # Successors (outgoing edges)
            for neighbor in self._graph.successors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

            # Predecessors (incoming edges)
            for neighbor in self._graph.predecessors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        subgraph = self._graph.subgraph(visited).copy()

        logger.info(
            f"  Subgraph extracted: {subgraph.number_of_nodes()} nodes, "
            f"{subgraph.number_of_edges()} edges"
        )

        return subgraph

    def trace_money_flow(
        self,
        source: str,
        max_depth: int = 5,
        min_amount: float = 0.0,
    ) -> List[List[Tuple[str, str, float]]]:
        """
        Trace money flow paths from a source account using DFS.

        Parameters
        ----------
        source : str
            Starting account ID.
        max_depth : int
            Maximum path length to trace.
        min_amount : float
            Minimum edge weight (transaction amount) to follow.

        Returns
        -------
        List[List[Tuple[str, str, float]]]
            List of paths, each path is a list of (from, to, amount) tuples.
        """
        if self._graph is None:
            raise RuntimeError("Graph not built. Call build_graph() first.")

        if source not in self._graph:
            logger.warning(f"Source node '{source}' not found in graph")
            return []

        logger.info(f"Tracing money flow from '{source}', max_depth={max_depth}")

        all_paths: List[List[Tuple[str, str, float]]] = []

        def dfs(node: str, current_path: List[Tuple[str, str, float]], visited: Set[str]) -> None:
            if len(current_path) >= max_depth:
                all_paths.append(current_path.copy())
                return

            has_successors = False
            for successor in self._graph.successors(node):
                edge_data = self._graph[node][successor]
                amount = edge_data.get("weight", 0)

                if amount < min_amount:
                    continue
                if successor in visited:
                    # Found a cycle in the path --- record it
                    all_paths.append(current_path + [(node, successor, amount)])
                    continue

                has_successors = True
                visited.add(successor)
                current_path.append((node, successor, amount))
                dfs(successor, current_path, visited)
                current_path.pop()
                visited.discard(successor)

            if not has_successors and current_path:
                all_paths.append(current_path.copy())

        dfs(source, [], {source})

        logger.info(f"  Found {len(all_paths)} flow paths from '{source}'")

        return all_paths

    def generate_graph_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate per-transaction graph-based features.

        For each transaction, extract features from the graph context of
        the sender and receiver accounts.

        Parameters
        ----------
        df : pd.DataFrame
            Transaction data with from_id and to_id columns.

        Returns
        -------
        pd.DataFrame
            Graph features aligned with the transaction dataframe.
        """
        if self._graph is None:
            raise RuntimeError("Graph not built. Call build_graph() first.")

        logger.info("Generating graph features for transactions")

        feats = pd.DataFrame(index=df.index)

        # Ensure centrality and communities are computed
        if self._centrality is None:
            self.compute_centrality()
        if self._communities is None:
            self.detect_communities()

        # Node-level features for sender
        for metric_name, values in self._centrality.items():
            sender_vals = df["from_id"].map(values).fillna(0)
            receiver_vals = df["to_id"].map(values).fillna(0)
            feats[f"graph_sender_{metric_name}"] = sender_vals.values
            feats[f"graph_receiver_{metric_name}"] = receiver_vals.values

        # Degree features
        for node_attr in ["out_degree", "in_degree", "total_out_flow", "total_in_flow", "net_flow"]:
            node_values = {
                node: data.get(node_attr, 0)
                for node, data in self._graph.nodes(data=True)
            }
            feats[f"graph_sender_{node_attr}"] = df["from_id"].map(node_values).fillna(0).values
            feats[f"graph_receiver_{node_attr}"] = df["to_id"].map(node_values).fillna(0).values

        # Community features
        if self._communities:
            sender_comm = df["from_id"].map(self._communities).fillna(-1).astype(int)
            receiver_comm = df["to_id"].map(self._communities).fillna(-1).astype(int)
            feats["graph_sender_community"] = sender_comm.values
            feats["graph_receiver_community"] = receiver_comm.values
            feats["graph_same_community"] = (sender_comm == receiver_comm).astype(int).values

            # Community size
            community_sizes = defaultdict(int)
            for node, comm in self._communities.items():
                community_sizes[comm] += 1
            feats["graph_sender_community_size"] = sender_comm.map(community_sizes).fillna(0).values
            feats["graph_receiver_community_size"] = receiver_comm.map(community_sizes).fillna(0).values

        # Cycle involvement
        cycle_nodes: Set[str] = set()
        for cycle in self._cycles:
            cycle_nodes.update(cycle)

        feats["graph_sender_in_cycle"] = df["from_id"].isin(cycle_nodes).astype(int).values
        feats["graph_receiver_in_cycle"] = df["to_id"].isin(cycle_nodes).astype(int).values

        # Edge features (if the edge exists in the graph)
        edge_txn_count = []
        edge_total_amount = []
        for _, row in df.iterrows():
            if self._graph.has_edge(row["from_id"], row["to_id"]):
                edge_data = self._graph[row["from_id"]][row["to_id"]]
                edge_txn_count.append(edge_data.get("txn_count", 0))
                edge_total_amount.append(edge_data.get("weight", 0))
            else:
                edge_txn_count.append(0)
                edge_total_amount.append(0)

        feats["graph_edge_txn_count"] = edge_txn_count
        feats["graph_edge_total_amount"] = edge_total_amount

        # Neighbor overlap (Jaccard similarity between sender and receiver neighborhoods)
        jaccard_scores = []
        for _, row in df.iterrows():
            sender = row["from_id"]
            receiver = row["to_id"]
            if sender in self._graph and receiver in self._graph:
                sender_neighbors = set(self._graph.successors(sender)) | set(self._graph.predecessors(sender))
                receiver_neighbors = set(self._graph.successors(receiver)) | set(self._graph.predecessors(receiver))
                intersection = sender_neighbors & receiver_neighbors
                union = sender_neighbors | receiver_neighbors
                jaccard = len(intersection) / len(union) if union else 0
            else:
                jaccard = 0
            jaccard_scores.append(jaccard)

        feats["graph_neighbor_jaccard"] = jaccard_scores

        logger.info(f"  Generated {feats.shape[1]} graph features")

        return feats

    def identify_suspicious_accounts(
        self,
        top_k: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Identify the most suspicious accounts based on graph analysis.

        Combines centrality, community, and cycle information into a
        composite graph suspiciousness score.

        Parameters
        ----------
        top_k : int, optional
            Return top K suspicious accounts. Defaults to config value.

        Returns
        -------
        pd.DataFrame
            Ranked list of suspicious accounts with scores.
        """
        if self._graph is None:
            raise RuntimeError("Graph not built. Call build_graph() first.")

        top_k = top_k or self.graph_config.top_k_central_nodes

        # Ensure metrics are computed
        if self._centrality is None:
            self.compute_centrality()
        if self._communities is None:
            self.detect_communities()
        if not self._cycles:
            self.detect_cycles()

        records = []
        cycle_nodes = set()
        for cycle in self._cycles:
            cycle_nodes.update(cycle)

        for node in self._graph.nodes():
            betweenness = self._centrality.get("betweenness", {}).get(node, 0)
            pagerank = self._centrality.get("pagerank", {}).get(node, 0)
            in_cycle = node in cycle_nodes
            community = self._communities.get(node, -1) if self._communities else -1
            node_data = self._graph.nodes[node]

            # Composite score
            score = (
                0.3 * min(1.0, betweenness * 100)
                + 0.3 * min(1.0, pagerank * 100)
                + 0.2 * (1.0 if in_cycle else 0.0)
                + 0.1 * min(1.0, node_data.get("out_degree", 0) / 50)
                + 0.1 * min(1.0, node_data.get("in_degree", 0) / 50)
            )

            records.append({
                "account_id": node,
                "graph_suspicion_score": score,
                "betweenness_centrality": betweenness,
                "pagerank": pagerank,
                "in_cycle": in_cycle,
                "community_id": community,
                "out_degree": node_data.get("out_degree", 0),
                "in_degree": node_data.get("in_degree", 0),
                "total_out_flow": node_data.get("total_out_flow", 0),
                "total_in_flow": node_data.get("total_in_flow", 0),
                "net_flow": node_data.get("net_flow", 0),
            })

        result = pd.DataFrame(records)
        result = result.sort_values("graph_suspicion_score", ascending=False).head(top_k)
        result = result.reset_index(drop=True)

        logger.info(f"  Top-{top_k} suspicious accounts identified by graph analysis")

        return result

    def get_graph_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of the transaction graph.

        Returns
        -------
        Dict[str, Any]
            Graph summary statistics.
        """
        if self._graph is None:
            return {"status": "Graph not built"}

        G = self._graph
        n_communities = len(set(self._communities.values())) if self._communities else 0

        return {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "density": nx.density(G),
            "is_weakly_connected": nx.is_weakly_connected(G) if G.number_of_nodes() > 0 else False,
            "weakly_connected_components": nx.number_weakly_connected_components(G),
            "strongly_connected_components": nx.number_strongly_connected_components(G),
            "communities_detected": n_communities,
            "cycles_detected": len(self._cycles),
            "avg_clustering": nx.average_clustering(G.to_undirected()) if G.number_of_nodes() > 0 else 0,
        }
