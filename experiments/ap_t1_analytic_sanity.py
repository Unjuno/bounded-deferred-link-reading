#!/usr/bin/env python3
"""AP-T1: exact analytic sanity checks for deferred-query theory.

This is not a confirmatory human/LM experiment. It verifies finite-state consequences
of the analytic model on small random precedence-constrained query trees.
"""
import itertools
import json
import random
from functools import lru_cache
from pathlib import Path

SEED = 20260827
N_TREES = 200

rng = random.Random(SEED)


def gen_tree(max_nodes=8):
    n = rng.randint(4, max_nodes)
    parents = [None, None]
    depths = [0, 0]
    for i in range(2, n):
        candidates = [j for j in range(i) if depths[j] < 3]
        p = rng.choice(candidates)
        parents.append(p)
        depths.append(depths[p] + 1)
    children = [[] for _ in range(n)]
    roots = []
    for i, p in enumerate(parents):
        if p is None:
            roots.append(i)
        else:
            children[p].append(i)
    costs = [rng.randint(1, 3) for _ in range(n)]
    rewards = [rng.uniform(0.2, 2.0) for _ in range(n)]
    return {"children": children, "roots": roots, "costs": costs, "rewards": rewards}


def optimal(tree, budget, k=None):
    children, costs, rewards = tree["children"], tree["costs"], tree["rewards"]
    roots = tuple(tree["roots"])
    init_sets = [roots] if k is None or len(roots) <= k else list(itertools.combinations(roots, k))

    @lru_cache(None)
    def value(frontier, b):
        frontier = list(frontier)
        best = 0.0
        for q in list(frontier):
            if costs[q] > b:
                continue
            new = [x for x in frontier if x != q] + children[q]
            if k is None or len(new) <= k:
                next_frontiers = [tuple(sorted(new))]
            else:
                next_frontiers = [tuple(sorted(s)) for s in itertools.combinations(new, k)]
            for nf in next_frontiers:
                best = max(best, rewards[q] + value(nf, b - costs[q]))
        return best

    return max(value(tuple(sorted(s)), budget) for s in init_sets)


def immediate_greedy(tree, budget):
    frontier = list(tree["roots"])
    total = 0.0
    while True:
        eligible = [q for q in frontier if tree["costs"][q] <= budget]
        if not eligible:
            break
        q = max(
            eligible,
            key=lambda q: (tree["rewards"][q] / tree["costs"][q], tree["rewards"][q]),
        )
        frontier.remove(q)
        budget -= tree["costs"][q]
        total += tree["rewards"][q]
        frontier.extend(tree["children"][q])
    return total


def main():
    trees = [gen_tree() for _ in range(N_TREES)]
    budget_monotone = True
    capacity_monotone = True
    oracle_ge_greedy = True
    saturation = True
    greedy_gaps = []
    k_gaps = {1: [], 2: [], 3: [], 4: []}

    for tree in trees:
        total_cost = sum(tree["costs"])
        values = [optimal(tree, b) for b in range(total_cost + 1)]
        budget_monotone &= all(values[i + 1] + 1e-10 >= values[i] for i in range(len(values) - 1))
        saturation &= abs(optimal(tree, total_cost) - sum(tree["rewards"])) < 1e-9
        saturation &= abs(immediate_greedy(tree, total_cost) - sum(tree["rewards"])) < 1e-9

        for budget in sorted(set([max(1, total_cost // 3), max(1, total_cost // 2), total_cost])):
            oracle = optimal(tree, budget)
            greedy = immediate_greedy(tree, budget)
            oracle_ge_greedy &= oracle + 1e-10 >= greedy
            if oracle > 0:
                greedy_gaps.append((oracle - greedy) / oracle)

            kvals = [optimal(tree, budget, k) for k in [1, 2, 3, 4]]
            capacity_monotone &= all(kvals[i + 1] + 1e-10 >= kvals[i] for i in range(3))
            for k, v in zip([1, 2, 3, 4], kvals):
                if oracle > 0:
                    k_gaps[k].append((oracle - v) / oracle)

    threshold_failures = 0
    threshold_checks = 0
    # One-step model:
    # ASK now = v-c
    # DEFER = v-(1-p)c-h
    # Therefore ASK iff h >= p*c.
    for p in [0, 0.1, 0.25, 0.5, 0.75, 1.0]:
        for c in [0.5, 1, 2, 4]:
            for h in [0, 0.25, 0.5, 1, 2, 4]:
                v = 5.0
                ask = v - c
                defer = v - (1 - p) * c - h
                predicted_ask = h >= p * c - 1e-12
                actual_ask = ask >= defer - 1e-12
                threshold_checks += 1
                threshold_failures += int(predicted_ask != actual_ask)

    result = {
        "name": "AP-T1 exact analytic sanity checks",
        "seed": SEED,
        "n_random_trees": N_TREES,
        "checks": {
            "budget_value_monotone": budget_monotone,
            "capacity_value_monotone_K1_to_K4": capacity_monotone,
            "full_oracle_ge_immediate_greedy": oracle_ge_greedy,
            "all_work_conserving_policies_saturate_at_total_cost": saturation,
            "single_unknown_threshold_checks": threshold_checks,
            "single_unknown_threshold_failures": threshold_failures,
        },
        "mean_relative_full_minus_immediate_greedy": sum(greedy_gaps) / len(greedy_gaps),
        "mean_relative_full_minus_K": {
            str(k): sum(v) / len(v) for k, v in k_gaps.items()
        },
        "interpretation": "Sanity check only: verifies finite-state consequences of the analytic model; not evidence about human comprehension or LM factuality.",
    }

    out = Path("results/AP_T1_ANALYTIC_SANITY.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
