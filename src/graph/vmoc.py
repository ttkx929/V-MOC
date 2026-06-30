from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def estimate_tokens(text: str) -> int:
    """Estimate token cost with a robust local fallback."""
    if not text:
        return 0
    try:
        import tiktoken

        encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(text))
    except Exception:
        return max(1, len(text.split()))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0])


@dataclass
class ValuePath:
    nodes: List[str]
    roles: List[str]
    messages: List[str]
    value: float
    cost: int
    reliability: float
    divergence: float
    novelty: float
    hop: int
    source_root: str
    embedding: np.ndarray


@dataclass
class ClusterMessage:
    hop: int
    node_id: str
    role: str
    content: str
    embedding: np.ndarray
    source_root: str
    original_idx: int
    source_path: List[str] = field(default_factory=list)

    @property
    def cost(self) -> int:
        return estimate_tokens(self.content)


class PathEvaluator:
    """Path-level value evaluation for V-MOC."""

    def __init__(
        self,
        graph,
        alpha: float,
        beta: float,
        gamma: float,
        lambda_cost: float,
        trust_decay: float,
        trust_threshold: float,
    ):
        self.graph = graph
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_cost = lambda_cost
        self.trust_decay = trust_decay
        self.trust_threshold = trust_threshold

    def enumerate_paths(self, target_id: str, max_hops: int, query: str) -> List[ValuePath]:
        if max_hops < 1:
            return []

        if target_id in self.graph.nodes:
            target = self.graph.nodes[target_id]
        elif getattr(self.graph.decision_node, "id", None) == target_id:
            target = self.graph.decision_node
        else:
            return []

        query_embedding = self.graph.encode_text(query)
        paths: List[List] = []

        def dfs(current, reverse_path: List, depth: int) -> None:
            if depth > max_hops:
                return
            for predecessor in current.spatial_predecessors:
                candidate = reverse_path + [predecessor]
                paths.append(list(reversed(candidate)))
                dfs(predecessor, candidate, depth + 1)

        dfs(target, [target], 1)

        value_paths: List[ValuePath] = []
        for path_nodes in paths:
            upstream_nodes = path_nodes[:-1]
            if not upstream_nodes:
                continue

            trust_product = 1.0
            messages: List[str] = []
            roles: List[str] = []
            node_ids: List[str] = []
            for node in upstream_nodes:
                trust = self.graph.trust_manager.get(node.id)
                if trust < self.trust_threshold:
                    trust_product = 0.0
                    break
                output = node.outputs[-1] if node.outputs else ""
                messages.append(str(output))
                roles.append(node.role)
                node_ids.append(node.id)
                trust_product *= trust

            if trust_product <= 0.0 or not any(messages):
                continue

            hop = len(upstream_nodes)
            path_text = "\n\n".join(messages)
            path_embedding = self.graph.encode_text(path_text)
            reliability = trust_product * (self.trust_decay ** hop)
            divergence = 1.0 - cosine(path_embedding, query_embedding)
            novelty = 1.0
            cost = sum(estimate_tokens(message) for message in messages)
            source_root = self.find_source_root(upstream_nodes[0].id)
            value = self.score(reliability, divergence, novelty, cost)
            value_paths.append(
                ValuePath(
                    nodes=node_ids + [target_id],
                    roles=roles,
                    messages=messages,
                    value=value,
                    cost=cost,
                    reliability=reliability,
                    divergence=divergence,
                    novelty=novelty,
                    hop=hop,
                    source_root=source_root,
                    embedding=path_embedding,
                )
            )

        return value_paths

    def find_source_root(self, node_id: str) -> str:
        node = self.graph.nodes[node_id]
        if not node.spatial_predecessors:
            return node_id

        visited: Set[str] = set()
        roots: List[str] = []

        def dfs(current) -> None:
            if current.id in visited:
                return
            visited.add(current.id)
            if not current.spatial_predecessors:
                roots.append(current.id)
                return
            for predecessor in current.spatial_predecessors:
                dfs(predecessor)

        dfs(node)
        if not roots:
            return node_id
        return min(roots, key=self.graph.topological_rank)

    def score(self, reliability: float, divergence: float, novelty: float, cost: int) -> float:
        return (
            self.alpha * reliability
            + self.beta * divergence
            + self.gamma * novelty
            - self.lambda_cost * cost
        )


class BudgetManager:
    """Per-agent token budget B_j with path selection and merge-step updates."""

    def __init__(self, token_budget: int):
        self.initial_budget = token_budget
        self.remaining_budget = token_budget

    def reset(self) -> None:
        self.remaining_budget = self.initial_budget

    def select(self, paths: List[ValuePath], evaluator: PathEvaluator) -> Tuple[List[ValuePath], int, int]:
        """Greedy path selection under current remaining budget (sum C(p) <= B_j)."""
        selected: List[ValuePath] = []
        total_cost = 0

        candidates = list(paths)
        while candidates:
            for path in candidates:
                if selected:
                    path.novelty = max(1.0 - cosine(path.embedding, chosen.embedding) for chosen in selected)
                else:
                    path.novelty = 1.0
                path.value = evaluator.score(path.reliability, path.divergence, path.novelty, path.cost)

            path = max(candidates, key=lambda item: item.value)
            if total_cost + path.cost <= self.remaining_budget:
                selected.append(path)
                total_cost += path.cost
            else:
                break
            candidates.remove(path)

        adaptive_k = max((path.hop for path in selected), default=0)
        return selected, adaptive_k, total_cost

    def update_after_merge(self, merged_input_tokens: int, compressed_tokens: int) -> int:
        """B_j^{(t+1)} = B_j^{(t)} - TokenCount(merged) + TokenCount(compressed)."""
        self.remaining_budget = self.remaining_budget - merged_input_tokens + compressed_tokens
        return self.remaining_budget


class AgentBudgetRegistry:
    """Maintains independent B_j pools for each target agent."""

    def __init__(self, token_budget: int, node_ids: Optional[Iterable[str]] = None):
        self.token_budget = token_budget
        self._managers: Dict[str, BudgetManager] = {}
        if node_ids is not None:
            self.ensure_nodes(node_ids)

    def ensure_nodes(self, node_ids: Iterable[str]) -> None:
        for node_id in node_ids:
            if node_id not in self._managers:
                self._managers[node_id] = BudgetManager(self.token_budget)

    def reset_round(self, node_ids: Iterable[str]) -> None:
        self.ensure_nodes(node_ids)
        for node_id in node_ids:
            self._managers[node_id].reset()

    def get(self, node_id: str) -> BudgetManager:
        self.ensure_nodes([node_id])
        return self._managers[node_id]


class ClusterMerger:
    """Path-aware semantic-topological merging."""

    def __init__(
        self,
        graph,
        epsilon: float,
        kppa: int,
        eta: float,
        nu: float,
        theta_cross: float,
    ):
        self.graph = graph
        self.epsilon = epsilon
        self.kppa = kppa
        self.eta = eta
        self.nu = nu
        self.theta_cross = theta_cross

    def collect_messages(self, paths: Sequence[ValuePath]) -> List[ClusterMessage]:
        collected: Dict[Tuple[str, str], ClusterMessage] = {}
        original_idx = 0
        for path in paths:
            upstream_ids = path.nodes[:-1]
            for node_id, role, content in zip(upstream_ids, path.roles, path.messages):
                key = (path.source_root, node_id)
                if key in collected:
                    continue
                collected[key] = ClusterMessage(
                    hop=path.hop,
                    node_id=node_id,
                    role=role,
                    content=content,
                    embedding=self.graph.encode_text(content),
                    source_root=path.source_root,
                    original_idx=original_idx,
                    source_path=path.nodes,
                )
                original_idx += 1
        return sorted(collected.values(), key=lambda item: item.original_idx)

    async def merge(self, messages: List[ClusterMessage]) -> str:
        if not messages:
            return ""

        clusters: Dict[str, List[ClusterMessage]] = {}
        for message in messages:
            clusters.setdefault(message.source_root, []).append(message)

        merged_messages: List[ClusterMessage] = []
        for cluster_messages in clusters.values():
            merged_messages.extend(await self.merge_cluster(cluster_messages, list(clusters.values())))

        merged_messages = await self.cross_cluster_merge(merged_messages)
        merged_messages.sort(key=lambda item: (-item.hop, item.original_idx))
        return "\n\n".join(
            f"Agent {item.node_id}, role is {item.role}, output is:\n{item.content}"
            for item in merged_messages
        )

    async def merge_cluster(
        self,
        messages: List[ClusterMessage],
        all_clusters: List[List[ClusterMessage]],
    ) -> List[ClusterMessage]:
        working = sorted(messages, key=lambda item: item.original_idx)
        other_embeddings = [
            msg.embedding
            for cluster in all_clusters
            if cluster is not messages
            for msg in cluster
        ]

        while len(working) > 1:
            vectors = np.array([item.embedding for item in working])
            sim_matrix = cosine_similarity(vectors)
            np.fill_diagonal(sim_matrix, -1)
            max_similarity = float(np.max(sim_matrix))
            threshold = max_similarity - self.epsilon

            candidates: List[Tuple[int, int, float]] = []
            for i in range(len(working)):
                for j in range(i + 1, len(working)):
                    if sim_matrix[i, j] >= threshold:
                        candidates.append((i, j, float(sim_matrix[i, j])))
            if not candidates:
                break

            candidates.sort(key=lambda item: item[2], reverse=True)
            used: Set[int] = set()
            selected: List[Tuple[int, int, float]] = []
            for i, j, sim in candidates:
                if i not in used and j not in used:
                    selected.append((i, j, sim))
                    used.update({i, j})

            if not selected:
                break

            new_working: List[ClusterMessage] = []
            merged_indices: Set[int] = set()
            for i, j, _ in selected:
                left = working[i]
                right = working[j]
                merged = await self.merge_pair(left, right, other_embeddings, preserve_clusters=False)
                new_working.append(merged)
                merged_indices.update({i, j})

            for idx, item in enumerate(working):
                if idx not in merged_indices:
                    new_working.append(item)
            working = sorted(new_working, key=lambda item: item.original_idx)

        return working

    async def cross_cluster_merge(self, messages: List[ClusterMessage]) -> List[ClusterMessage]:
        working = sorted(messages, key=lambda item: item.original_idx)
        changed = True
        while changed and len(working) > 1:
            changed = False
            for i in range(len(working)):
                if changed:
                    break
                for j in range(i + 1, len(working)):
                    if working[i].source_root == working[j].source_root:
                        continue
                    if cosine(working[i].embedding, working[j].embedding) > self.theta_cross:
                        other_embeddings = [
                            item.embedding
                            for idx, item in enumerate(working)
                            if idx not in {i, j}
                        ]
                        merged = await self.merge_pair(
                            working[i],
                            working[j],
                            other_embeddings,
                            preserve_clusters=True,
                        )
                        working = [
                            item for idx, item in enumerate(working) if idx not in {i, j}
                        ] + [merged]
                        working.sort(key=lambda item: item.original_idx)
                        changed = True
                        break
        return working

    async def merge_pair(
        self,
        left: ClusterMessage,
        right: ClusterMessage,
        other_embeddings: Sequence[np.ndarray],
        preserve_clusters: bool,
    ) -> ClusterMessage:
        content = await self.graph.merge_multiple_messages_value_aware(
            [left.node_id, left.role, left.content, right.node_id, right.role, right.content],
            self.kppa,
            other_embeddings=other_embeddings,
            eta=self.eta,
            nu=self.nu,
        )
        source_root = (
            f"{left.source_root} & {right.source_root}"
            if preserve_clusters and left.source_root != right.source_root
            else right.source_root
        )
        return ClusterMessage(
            hop=min(left.hop, right.hop),
            node_id=f"{left.node_id} & {right.node_id}",
            role=f"{left.role} & {right.role}",
            content=content,
            embedding=self.graph.encode_text(content),
            source_root=source_root,
            original_idx=max(left.original_idx, right.original_idx),
            source_path=left.source_path + right.source_path,
        )


class TrustManager:
    """Closed-loop trust update and cross-sample memory."""

    def __init__(self, node_ids: Iterable[str], initial: float = 0.5):
        self.trust: Dict[str, float] = {node_id: initial for node_id in node_ids}
        self.history_contributions: Dict[str, List[float]] = {
            node_id: [] for node_id in node_ids
        }
        self.last_contributors: Dict[str, np.ndarray] = {}
        self.last_final_embedding: Optional[np.ndarray] = None
        self.last_contexts: List[str] = []
        self.rho = 0.2
        self.delta = 0.01

    def ensure_nodes(self, node_ids: Iterable[str]) -> None:
        for node_id in node_ids:
            self.trust.setdefault(node_id, 0.5)
            self.history_contributions.setdefault(node_id, [])

    def get(self, node_id: str) -> float:
        self.ensure_nodes([node_id])
        return self.trust[node_id]

    def start_round(self) -> None:
        self.last_contributors = {}
        self.last_final_embedding = None
        self.last_contexts = []

    def record_messages(self, messages: Sequence[ClusterMessage], final_context: str, embed_fn) -> None:
        for message in messages:
            if " & " not in message.node_id:
                self.last_contributors[message.node_id] = message.embedding
        if final_context:
            self.last_contexts.append(final_context)
            self.last_final_embedding = embed_fn("\n\n".join(self.last_contexts))

    def update_with_feedback(self, is_correct: bool) -> None:
        if self.last_final_embedding is None:
            return

        contributor_ids = set(self.last_contributors.keys())
        for node_id, embedding in self.last_contributors.items():
            cos_i = cosine(embedding, self.last_final_embedding)
            others = [
                cosine(other, self.last_final_embedding)
                for other_id, other in self.last_contributors.items()
                if other_id != node_id
            ]
            avg_except = float(np.mean(others)) if others else 0.0
            contribution = max(0.0, cos_i - avg_except)
            contribution *= 1.0 if is_correct else 0.3
            self.trust[node_id] = (1 - self.rho) * self.get(node_id) + self.rho * contribution
            self.history_contributions[node_id].append(contribution)

        for node_id in list(self.trust.keys()):
            if node_id not in contributor_ids:
                self.trust[node_id] = self.trust[node_id] * (1 - self.delta)
                self.history_contributions.setdefault(node_id, []).append(0.0)

    def merge_from(self, other: "TrustManager") -> None:
        self.ensure_nodes(other.trust.keys())
        for node_id, value in other.trust.items():
            self.trust[node_id] = (1 - self.rho) * self.get(node_id) + self.rho * value
            self.history_contributions.setdefault(node_id, []).extend(
                other.history_contributions.get(node_id, [])[-1:]
            )
