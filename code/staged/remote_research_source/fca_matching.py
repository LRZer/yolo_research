"""Pure-Python bipartite matching helpers for FCA-P0.

The functions in this module deliberately have no torch/Ultralytics dependency so
their exact cardinality semantics can be unit-tested on CPU before any GPU probe.
"""

from __future__ import annotations

from collections import deque
import math
from typing import Hashable, Iterable, Mapping


Left = Hashable
Right = Hashable


def _sorted(values):
    """Natural ordering for homogeneous numeric IDs, repr fallback for mixed hashables."""
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=repr)


def canonical_adjacency(adjacency: Mapping[Left, Iterable[Right]]) -> dict[Left, tuple[Right, ...]]:
    """Return a deterministic, duplicate-free adjacency mapping."""
    return {left: tuple(_sorted(set(adjacency[left]))) for left in _sorted(adjacency)}


def validate_matching(adjacency: Mapping[Left, Iterable[Right]], matching: Mapping[Left, Right]) -> None:
    """Raise if a purported left->right matching contains a missing or repeated edge."""
    graph = canonical_adjacency(adjacency)
    if len(set(matching.values())) != len(matching):
        raise AssertionError("matching repeats a right node")
    for left, right in matching.items():
        if left not in graph or right not in graph[left]:
            raise AssertionError(f"matching edge {(left, right)!r} is absent from the graph")


def maximum_matching(adjacency: Mapping[Left, Iterable[Right]]) -> dict[Left, Right]:
    """Exact Hopcroft-Karp maximum-cardinality matching, returned as left->right."""
    graph = canonical_adjacency(adjacency)
    left_nodes = tuple(graph)
    pair_left: dict[Left, Right] = {}
    pair_right: dict[Right, Left] = {}
    distance: dict[Left, int] = {}
    infinity = len(left_nodes) + 1

    def bfs() -> bool:
        queue: deque[Left] = deque()
        shortest = infinity
        for left in left_nodes:
            if left not in pair_left:
                distance[left] = 0
                queue.append(left)
            else:
                distance[left] = infinity
        while queue:
            left = queue.popleft()
            if distance[left] >= shortest:
                continue
            for right in graph[left]:
                mate = pair_right.get(right)
                if mate is None:
                    shortest = distance[left] + 1
                elif distance[mate] == infinity:
                    distance[mate] = distance[left] + 1
                    queue.append(mate)
        return shortest != infinity

    def dfs(left: Left) -> bool:
        for right in graph[left]:
            mate = pair_right.get(right)
            if mate is None or (distance.get(mate, infinity) == distance[left] + 1 and dfs(mate)):
                pair_left[left] = right
                pair_right[right] = left
                return True
        distance[left] = infinity
        return False

    while bfs():
        for left in left_nodes:
            if left not in pair_left:
                dfs(left)
    validate_matching(graph, pair_left)
    return pair_left


def extend_matching_preserving_left_coverage(
    adjacency: Mapping[Left, Iterable[Right]], seed: Mapping[Left, Right]
) -> dict[Left, Right]:
    """Augment a valid seed matching to maximum cardinality without un-covering seed left nodes.

    An augmenting path can change a seed node's anchor, but every left node that is
    covered by ``seed`` remains covered.  This gives an exact, additive attribution
    of ``|M*|-|seed|`` newly coverable GTs for subgroup and zero-quality summaries.
    """
    graph = canonical_adjacency(adjacency)
    pair_left: dict[Left, Right] = {}
    pair_right: dict[Right, Left] = {}
    for left, right in sorted(seed.items(), key=lambda x: repr(x[0])):
        if left not in graph or right not in graph[left]:
            raise ValueError(f"seed edge {(left, right)!r} is absent from the graph")
        if right in pair_right:
            raise ValueError(f"seed is not a matching: right node {right!r} is repeated")
        pair_left[left] = right
        pair_right[right] = left

    def augment(left: Left, seen_left: set[Left], seen_right: set[Right]) -> bool:
        if left in seen_left:
            return False
        seen_left.add(left)
        for right in graph[left]:
            if right in seen_right:
                continue
            seen_right.add(right)
            mate = pair_right.get(right)
            if mate is None or augment(mate, seen_left, seen_right):
                pair_left[left] = right
                pair_right[right] = left
                return True
        return False

    while True:
        progress = False
        for left in graph:
            if left not in pair_left and augment(left, set(), set()):
                progress = True
        if not progress:
            break
    optimum = maximum_matching(graph)
    if len(pair_left) != len(optimum):
        raise AssertionError(f"seed augmentation is not maximum: got {len(pair_left)}, expected {len(optimum)}")
    if not set(seed).issubset(pair_left):
        raise AssertionError("seed-covered left nodes were lost during augmentation")
    validate_matching(graph, pair_left)
    return pair_left


def maximum_matching_preserving_left_set(
    adjacency: Mapping[Left, Iterable[Right]], required_left: Iterable[Left]
) -> tuple[dict[Left, Right], dict[Left, Right], bool]:
    """Find a maximum matching while preserving a requested set of covered left nodes.

    The native TAL conflict step can rewire an anchor to a GT that did not own that
    edge in the pre-conflict candidate graph.  Therefore, native *edges* are not a
    valid seed for that graph.  This helper first rematches the identities of all
    native-covered left nodes inside the candidate graph, then augments that seed
    to maximum cardinality.  The boolean is false when those identities cannot all
    be covered simultaneously; callers can still retain the returned unrestricted
    maximum for descriptive cardinality statistics, but must not claim additive
    no-sacrifice attribution in that case.
    """
    graph = canonical_adjacency(adjacency)
    required = set(required_left)
    missing = required - set(graph)
    if missing:
        return maximum_matching(graph), {}, False
    required_graph = {left: graph[left] for left in required}
    seed = maximum_matching(required_graph)
    if len(seed) != len(required):
        return maximum_matching(graph), seed, False
    maximum = extend_matching_preserving_left_coverage(graph, seed)
    return maximum, seed, True


def recoverable_gap_excluding_uncovered_left(
    adjacency: Mapping[Left, Iterable[Right]], seed: Mapping[Left, Right], excluded: Iterable[Left]
) -> int:
    """Gap after excluding selected *uncovered* left nodes while preserving every seed-covered node.

    Keeping excluded nodes that are already in ``seed`` prevents their anchors from
    being spuriously released. FCA-P0 uses this for objective zero-quality gap
    attribution: only zero-quality GTs that native S3 failed to cover are removed.
    """
    excluded_set = set(excluded) - set(seed)
    restricted = {left: rights for left, rights in adjacency.items() if left not in excluded_set}
    extended = extend_matching_preserving_left_coverage(restricted, seed)
    return len(extended) - len(seed)


def connected_components(adjacency: Mapping[Left, Iterable[Right]]) -> list[tuple[set[Left], set[Right]]]:
    """Connected components of a bipartite graph, excluding left nodes with no edges."""
    graph = canonical_adjacency(adjacency)
    reverse: dict[Right, set[Left]] = {}
    for left, rights in graph.items():
        for right in rights:
            reverse.setdefault(right, set()).add(left)
    unseen = {left for left, rights in graph.items() if rights}
    components: list[tuple[set[Left], set[Right]]] = []
    while unseen:
        first = _sorted(unseen)[0]
        lefts: set[Left] = set()
        rights: set[Right] = set()
        queue: deque[tuple[str, Hashable]] = deque([("L", first)])
        while queue:
            side, node = queue.popleft()
            if side == "L":
                if node in lefts:
                    continue
                lefts.add(node)
                unseen.discard(node)
                queue.extend(("R", right) for right in graph[node])
            else:
                if node in rights:
                    continue
                rights.add(node)
                queue.extend(("L", left) for left in reverse.get(node, ()))
        components.append((lefts, rights))
    components.sort(key=lambda item: repr(_sorted(item[0])[0]))
    return components


def lexicographic_quality_matching(
    adjacency: Mapping[Left, Iterable[Right]], weights: Mapping[tuple[Left, Right], float]
) -> dict[Left, Right]:
    """Maximize cardinality first, then total edge weight, using NetworkX exactly."""
    try:
        import networkx as nx
    except ImportError as error:  # pragma: no cover - production preflight requires NetworkX
        raise RuntimeError("FCA-P0 quality matching requires the preflighted NetworkX package") from error

    graph = nx.Graph()
    canonical = canonical_adjacency(adjacency)
    for left, rights in canonical.items():
        left_node = ("gt", left)
        graph.add_node(left_node, bipartite=0)
        for right in rights:
            if (left, right) not in weights:
                raise ValueError(f"missing quality weight for edge {(left, right)!r}")
            weight = float(weights[(left, right)])
            if not math.isfinite(weight) or weight < 0.0 or weight > 1.0 + 1e-12:
                raise ValueError(f"invalid normalized quality weight {weight!r} for edge {(left, right)!r}")
            right_node = ("anchor", right)
            graph.add_node(right_node, bipartite=1)
            graph.add_edge(left_node, right_node, weight=weight)
    raw = nx.algorithms.matching.max_weight_matching(graph, maxcardinality=True, weight="weight")
    result: dict[Left, Right] = {}
    for first, second in raw:
        if first[0] == "gt":
            left, right = first[1], second[1]
        else:
            left, right = second[1], first[1]
        result[left] = right
    expected = len(maximum_matching(adjacency))
    if len(result) != expected:
        raise AssertionError(f"weighted matching cardinality {len(result)} != exact maximum {expected}")
    validate_matching(canonical, result)
    return result
