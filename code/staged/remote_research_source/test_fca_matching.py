from __future__ import annotations

import itertools
import math
import random
import unittest

from fca_matching import (
    connected_components,
    extend_matching_preserving_left_coverage,
    lexicographic_quality_matching,
    maximum_matching,
    maximum_matching_preserving_left_set,
    recoverable_gap_excluding_uncovered_left,
)


def brute_force(adjacency):
    lefts = list(adjacency)
    best = {}

    def visit(index, used, current):
        nonlocal best
        if index == len(lefts):
            if len(current) > len(best):
                best = dict(current)
            return
        left = lefts[index]
        visit(index + 1, used, current)
        for right in adjacency[left]:
            if right not in used:
                used.add(right)
                current[left] = right
                visit(index + 1, used, current)
                current.pop(left)
                used.remove(right)

    visit(0, set(), {})
    return best


class MatchingTests(unittest.TestCase):
    def test_synthetic_native_gap(self):
        graph = {0: [10, 11], 1: [10]}
        native = {0: 10}
        optimum = maximum_matching(graph)
        extended = extend_matching_preserving_left_coverage(graph, native)
        self.assertEqual(len(optimum), 2)
        self.assertEqual(len(extended), 2)
        self.assertIn(0, extended)
        self.assertEqual(set(extended) - set(native), {1})

    def test_random_graphs_match_bruteforce(self):
        rng = random.Random(20260919)
        for n_left, n_right in itertools.product(range(1, 7), range(1, 7)):
            for _ in range(40):
                graph = {
                    left: [right for right in range(n_right) if rng.random() < 0.38]
                    for left in range(n_left)
                }
                expected = len(brute_force(graph))
                self.assertEqual(len(maximum_matching(graph)), expected)
                seed = {}
                used = set()
                for left in range(n_left):
                    choices = [right for right in graph[left] if right not in used]
                    if choices and rng.random() < 0.45:
                        seed[left] = choices[0]
                        used.add(choices[0])
                extended = extend_matching_preserving_left_coverage(graph, seed)
                self.assertEqual(len(extended), expected)
                self.assertTrue(set(seed).issubset(extended))

    def test_weighted_is_max_cardinality_and_max_weight(self):
        graph = {0: [0, 1], 1: [0, 1], 2: [1, 2]}
        weights = {
            (0, 0): 0.1,
            (0, 1): 0.9,
            (1, 0): 0.8,
            (1, 1): 0.2,
            (2, 1): 0.3,
            (2, 2): 0.7,
        }
        result = lexicographic_quality_matching(graph, weights)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(sum(weights[item] for item in result.items()), 2.4)

    def test_random_weighted_matches_bruteforce_objective(self):
        rng = random.Random(19092026)
        for n_left, n_right in itertools.product(range(1, 6), range(1, 6)):
            for _ in range(20):
                graph = {
                    left: [right for right in range(n_right) if rng.random() < 0.50]
                    for left in range(n_left)
                }
                weights = {
                    (left, right): rng.randrange(0, 101) / 100.0
                    for left, rights in graph.items()
                    for right in rights
                }
                expected_cardinality = len(brute_force(graph))
                best_weight = -1.0

                def visit(index, used, current_weight, cardinality):
                    nonlocal best_weight
                    if index == n_left:
                        if cardinality == expected_cardinality:
                            best_weight = max(best_weight, current_weight)
                        return
                    visit(index + 1, used, current_weight, cardinality)
                    for right in graph[index]:
                        if right not in used:
                            used.add(right)
                            visit(index + 1, used, current_weight + weights[(index, right)], cardinality + 1)
                            used.remove(right)

                visit(0, set(), 0.0, 0)
                actual = lexicographic_quality_matching(graph, weights)
                self.assertEqual(len(actual), expected_cardinality)
                self.assertAlmostEqual(sum(weights[item] for item in actual.items()), best_weight, places=9)

    def test_components(self):
        components = connected_components({0: [0], 1: [1, 2], 2: [2], 3: []})
        canonical = {(frozenset(lefts), frozenset(rights)) for lefts, rights in components}
        self.assertEqual(canonical, {(frozenset({0}), frozenset({0})), (frozenset({1, 2}), frozenset({1, 2}))})

    def test_bad_seed_is_rejected(self):
        with self.assertRaises(ValueError):
            extend_matching_preserving_left_coverage({0: [0], 1: [0]}, {0: 0, 1: 0})

    def test_empty_isolated_duplicate_and_disconnected(self):
        self.assertEqual(maximum_matching({}), {})
        graph = {0: [], 1: [4, 4], 2: [7], 3: [7, 8]}
        matching = maximum_matching(graph)
        self.assertEqual(len(matching), 3)
        components = connected_components(graph)
        self.assertEqual(sum(len(maximum_matching({g: graph[g] for g in lefts})) for lefts, _ in components), 3)

    def test_weight_validation(self):
        for bad in (float("nan"), float("inf"), -0.1, 1.1):
            with self.assertRaises(ValueError):
                lexicographic_quality_matching({0: [0]}, {(0, 0): bad})
        with self.assertRaises(ValueError):
            lexicographic_quality_matching({0: [0]}, {})

    def test_zero_quality_exclusion_does_not_release_native_anchor(self):
        # z0 and z2 are zero-quality. Native covers z0:a and n1:b. The only
        # full-graph gap unit is z2:c; deleting covered z0 would falsely free a
        # and make n3 look recoverable, masking the zero-quality contribution.
        graph = {0: [0], 1: [1], 2: [2], 3: [0]}
        native = {0: 0, 1: 1}
        all_gap = len(extend_matching_preserving_left_coverage(graph, native)) - len(native)
        nonzero_gap = recoverable_gap_excluding_uncovered_left(graph, native, {0, 2})
        self.assertEqual(all_gap, 1)
        self.assertEqual(nonzero_gap, 0)
        self.assertEqual(all_gap - nonzero_gap, 1)

    def test_native_identity_rematch_handles_offgraph_native_pairs(self):
        # Native pair identities may be off-graph, but both covered GTs can be
        # rematched in E_topk and then augmented without sacrificing coverage.
        graph = {0: [0, 1], 1: [1], 2: [0, 2]}
        maximum, seed, feasible = maximum_matching_preserving_left_set(graph, {0, 1})
        self.assertTrue(feasible)
        self.assertEqual(set(seed), {0, 1})
        self.assertEqual(set(maximum), {0, 1, 2})
        self.assertEqual(len(maximum) - len(seed), 1)

    def test_native_identity_hall_deficit_is_reported_without_false_witness(self):
        graph = {0: [0], 1: [0], 2: [1]}
        maximum, seed, feasible = maximum_matching_preserving_left_set(graph, {0, 1})
        self.assertFalse(feasible)
        self.assertEqual(len(seed), 1)
        self.assertEqual(len(maximum), 2)


if __name__ == "__main__":
    unittest.main()
