from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
FIT_SEED = 26082601
FIT_EPISODES = 1200
TEST_SEEDS = tuple(range(910001, 910013))
EPISODES_PER_SEED = 300
BUDGETS = (16, 32)
PRIMARY_BUDGET = 16
LONG_BUDGET = 32
K_VALUES = (1, 2, 4, 8, 16)
N_BOOT = 10000
RIDGE_ALPHA = 10.0
MAX_DEPTH = 4
N_ROOTS = 6
RHO = 0.92

@dataclass
class Node:
    node_id: int
    depth: int
    latent_quality: float
    reward: float
    latency: int
    cue: float
    children: List[int]
    parent_id: Optional[int]

def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)

def generate_recursive_query_tree(rng: np.random.Generator) -> Tuple[Dict[int, Node], List[int]]:
    """One base document whose query answers can reveal further unresolved spans."""
    nodes: Dict[int, Node] = {}
    roots: List[int] = []
    next_id = 0
    def build(depth: int, latent_quality: float, parent_id: Optional[int]) -> int:
        nonlocal next_id
        node_id = next_id
        next_id += 1
        reward = float(math.exp(0.85 * latent_quality + rng.normal(0.0, 0.25)))
        latency = int(np.clip(np.rint(math.exp(rng.normal(0.72, 0.45))), 1, 5))
        cue = float(latent_quality + rng.normal(0.0, 1.45))
        node = Node(node_id=node_id, depth=depth, latent_quality=float(latent_quality), reward=reward, latency=latency, cue=cue, children=[], parent_id=parent_id)
        nodes[node_id] = node
        if depth < MAX_DEPTH:
            lam = 1.0 + 0.6 * sigmoid(latent_quality)
            child_count = min(3, int(rng.poisson(lam)))
            for _ in range(child_count):
                child_quality = RHO * latent_quality + math.sqrt(1.0 - RHO * RHO) * float(rng.normal())
                child_id = build(depth + 1, child_quality, node_id)
                node.children.append(child_id)
        return node_id
    for _ in range(N_ROOTS):
        roots.append(build(0, float(rng.normal()), None))
    return nodes, roots

def compute_node_dps(nodes: Dict[int, Node], max_budget: int) -> Dict[int, np.ndarray]:
    """Exact ancestor-closed subtree knapsack DP for a clairvoyant serial-latency oracle."""
    cache: Dict[int, np.ndarray] = {}
    def solve(node_id: int) -> np.ndarray:
        if node_id in cache:
            return cache[node_id]
        node = nodes[node_id]
        out = np.full(max_budget + 1, -np.inf, dtype=float)
        out[0] = 0.0
        if node.latency <= max_budget:
            acc = np.full(max_budget + 1, -np.inf, dtype=float)
            acc[node.latency] = node.reward
            for child_id in node.children:
                child = solve(child_id)
                nxt = np.full(max_budget + 1, -np.inf, dtype=float)
                for b in range(max_budget + 1):
                    if not np.isfinite(acc[b]):
                        continue
                    for cb in range(max_budget - b + 1):
                        if np.isfinite(child[cb]):
                            nxt[b + cb] = max(nxt[b + cb], acc[b] + child[cb])
                acc = np.maximum(acc, nxt)
            out = np.maximum(out, acc)
        cache[node_id] = out
        return out
    for node_id in nodes:
        solve(node_id)
    return cache

def forest_oracle(nodes: Dict[int, Node], roots: Sequence[int], budget: int, node_dps: Dict[int, np.ndarray]) -> float:
    dp = np.full(budget + 1, -np.inf, dtype=float)
    dp[0] = 0.0
    for root_id in roots:
        root = node_dps[root_id][:budget + 1]
        nxt = np.full(budget + 1, -np.inf, dtype=float)
        for b in range(budget + 1):
            if not np.isfinite(dp[b]):
                continue
            for rb in range(budget - b + 1):
                if np.isfinite(root[rb]):
                    nxt[b + rb] = max(nxt[b + rb], dp[b] + root[rb])
        dp = nxt
    return float(np.max(dp))

def cue_rank(nodes: Dict[int, Node], node_id: int, frontier: Sequence[int]) -> float:
    cue = nodes[node_id].cue
    rank = 1 + sum(nodes[other].cue > cue for other in frontier)
    return rank / max(1, len(frontier))

def features_aware(nodes: Dict[int, Node], node_id: int, remaining: int, budget: int, frontier: Sequence[int]) -> np.ndarray:
    node = nodes[node_id]
    parent_reward = nodes[node.parent_id].reward if node.parent_id is not None else 1.0
    return np.array([node.cue, math.log1p(node.latency), node.depth / MAX_DEPTH, math.log1p(parent_reward), remaining / budget, math.log1p(len(frontier)), node.cue / node.latency, cue_rank(nodes, node_id, frontier)], dtype=float)

def features_no_latency(nodes: Dict[int, Node], node_id: int, remaining: int, budget: int, frontier: Sequence[int]) -> np.ndarray:
    node = nodes[node_id]
    parent_reward = nodes[node.parent_id].reward if node.parent_id is not None else 1.0
    return np.array([node.cue, node.depth / MAX_DEPTH, math.log1p(parent_reward), remaining / budget, math.log1p(len(frontier)), cue_rank(nodes, node_id, frontier)], dtype=float)

def greedy_choice(nodes: Dict[int, Node], frontier: Sequence[int], remaining: int) -> Optional[int]:
    eligible = [node_id for node_id in frontier if nodes[node_id].latency <= remaining]
    if not eligible:
        return None
    return max(eligible, key=lambda node_id: (nodes[node_id].cue / nodes[node_id].latency, nodes[node_id].cue, -nodes[node_id].latency))

def collect_teacher_rows(nodes: Dict[int, Node], roots: Sequence[int], budget: int, node_dps: Dict[int, np.ndarray], feature_fn) -> Tuple[List[np.ndarray], List[float]]:
    remaining = budget
    frontier = list(roots)
    xs: List[np.ndarray] = []
    ys: List[float] = []
    while True:
        eligible = [node_id for node_id in frontier if nodes[node_id].latency <= remaining]
        if not eligible:
            break
        for node_id in eligible:
            xs.append(feature_fn(nodes, node_id, remaining, budget, frontier))
            ys.append(float(np.max(node_dps[node_id][:remaining + 1])))
        chosen = greedy_choice(nodes, frontier, remaining)
        if chosen is None:
            break
        frontier.remove(chosen)
        node = nodes[chosen]
        remaining -= node.latency
        frontier.extend(node.children)
    return xs, ys

def train_models() -> Tuple[object, object, int]:
    rng = np.random.default_rng(FIT_SEED)
    xa: List[np.ndarray] = []
    xn: List[np.ndarray] = []
    y: List[float] = []
    max_budget = max(BUDGETS)
    for episode in range(FIT_EPISODES):
        nodes, roots = generate_recursive_query_tree(rng)
        dps = compute_node_dps(nodes, max_budget)
        for budget in BUDGETS:
            rows_a, target = collect_teacher_rows(nodes, roots, budget, dps, features_aware)
            rows_n, target_n = collect_teacher_rows(nodes, roots, budget, dps, features_no_latency)
            if target != target_n:
                raise RuntimeError('teacher target mismatch')
            xa.extend(rows_a)
            xn.extend(rows_n)
            y.extend(target)
        if episode % 300 == 0:
            print('fit_episode', episode, 'rows', len(y))
    model_aware = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    model_no_latency = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    model_aware.fit(np.vstack(xa), np.asarray(y))
    model_no_latency.fit(np.vstack(xn), np.asarray(y))
    return model_aware, model_no_latency, len(y)

def predicted_values(model, feature_fn, nodes, frontier, remaining, budget, candidates):
    x = np.vstack([feature_fn(nodes, node_id, remaining, budget, frontier) for node_id in candidates])
    return model.predict(x)

def compress_frontier(model, feature_fn, nodes, frontier, remaining, budget, k):
    if k is None or len(frontier) <= k:
        return list(frontier)
    pred = predicted_values(model, feature_fn, nodes, frontier, remaining, budget, frontier)
    order = np.argsort(-pred)
    return [frontier[int(i)] for i in order[:k]]

def rollout_learned(nodes, roots, budget, model, feature_fn, k=None) -> float:
    remaining = budget
    total = 0.0
    frontier = compress_frontier(model, feature_fn, nodes, list(roots), remaining, budget, k)
    while True:
        eligible = [node_id for node_id in frontier if nodes[node_id].latency <= remaining]
        if not eligible:
            break
        pred = predicted_values(model, feature_fn, nodes, frontier, remaining, budget, eligible)
        chosen = eligible[int(np.argmax(pred))]
        frontier.remove(chosen)
        node = nodes[chosen]
        remaining -= node.latency
        total += node.reward
        frontier.extend(node.children)
        frontier = compress_frontier(model, feature_fn, nodes, frontier, remaining, budget, k)
    return total

def rollout_greedy(nodes, roots, budget) -> float:
    remaining = budget
    total = 0.0
    frontier = list(roots)
    while True:
        chosen = greedy_choice(nodes, frontier, remaining)
        if chosen is None:
            break
        frontier.remove(chosen)
        node = nodes[chosen]
        remaining -= node.latency
        total += node.reward
        frontier.extend(node.children)
    return total

def rollout_recursive_dfs(nodes, roots, budget) -> float:
    remaining = budget
    total = 0.0
    frontier = sorted(roots, key=lambda node_id: nodes[node_id].cue / nodes[node_id].latency, reverse=True)
    while True:
        chosen = next((node_id for node_id in frontier if nodes[node_id].latency <= remaining), None)
        if chosen is None:
            break
        frontier.remove(chosen)
        node = nodes[chosen]
        remaining -= node.latency
        total += node.reward
        children = sorted(node.children, key=lambda node_id: nodes[node_id].cue / nodes[node_id].latency, reverse=True)
        frontier = children + frontier
    return total

def bootstrap_seed_ci(seed_means: np.ndarray, rng: np.random.Generator) -> Tuple[float, float]:
    n = len(seed_means)
    boots = np.empty(N_BOOT, dtype=float)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(np.mean(seed_means[idx]))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(lo), float(hi)

def summarize_by_seed(rows, policy, budget):
    out = []
    for seed in TEST_SEEDS:
        vals = [r['oracle_fraction'] for r in rows if r['seed'] == seed and r['policy'] == policy and r['budget'] == budget]
        out.append(float(np.mean(vals)))
    return np.asarray(out)

def run_confirmatory(model_aware, model_no_latency):
    rows = []
    max_budget = max(BUDGETS)
    for seed in TEST_SEEDS:
        rng = np.random.default_rng(seed)
        for episode in range(EPISODES_PER_SEED):
            nodes, roots = generate_recursive_query_tree(rng)
            dps = compute_node_dps(nodes, max_budget)
            for budget in BUDGETS:
                oracle_reward = forest_oracle(nodes, roots, budget, dps)
                if oracle_reward <= 0:
                    raise RuntimeError('non-positive oracle reward')
                policy_rewards = {
                    'recursive_dfs': rollout_recursive_dfs(nodes, roots, budget),
                    'greedy_visible': rollout_greedy(nodes, roots, budget),
                    'learned_full_aware': rollout_learned(nodes, roots, budget, model_aware, features_aware, None),
                    'learned_full_no_latency': rollout_learned(nodes, roots, budget, model_no_latency, features_no_latency, None),
                }
                for k in K_VALUES:
                    policy_rewards[f'learned_k{k}_aware'] = rollout_learned(nodes, roots, budget, model_aware, features_aware, k)
                for policy, reward in policy_rewards.items():
                    rows.append({'seed': seed, 'episode': episode, 'budget': budget, 'policy': policy, 'reward': reward, 'oracle': oracle_reward, 'oracle_fraction': reward / oracle_reward})
        print('test_seed_done', seed)
    return rows

def make_results(rows, teacher_rows):
    boot_rng = np.random.default_rng(880021)
    policies = sorted({r['policy'] for r in rows})
    aggregate = {}
    for budget in BUDGETS:
        aggregate[str(budget)] = {}
        for policy in policies:
            vals = np.asarray([r['oracle_fraction'] for r in rows if r['budget'] == budget and r['policy'] == policy])
            aggregate[str(budget)][policy] = {
                'mean_oracle_fraction': float(np.mean(vals)),
                'mean_reward': float(np.mean([r['reward'] for r in rows if r['budget'] == budget and r['policy'] == policy])),
            }
    def delta_ci(policy_a, policy_b, budget):
        a = summarize_by_seed(rows, policy_a, budget)
        b = summarize_by_seed(rows, policy_b, budget)
        d = a - b
        return {'mean': float(np.mean(d)), 'ci95': list(bootstrap_seed_ci(d, boot_rng)), 'positive_seeds': int(np.sum(d > 0)), 'seed_deltas': d.tolist()}
    h1 = delta_ci('learned_full_aware', 'greedy_visible', PRIMARY_BUDGET)
    h1_long = delta_ci('learned_full_aware', 'greedy_visible', LONG_BUDGET)
    latency = delta_ci('learned_full_aware', 'learned_full_no_latency', PRIMARY_BUDGET)
    latency_long = delta_ci('learned_full_aware', 'learned_full_no_latency', LONG_BUDGET)
    capacity = {}
    minimal_k = None
    for k in K_VALUES:
        gaps = {}
        ok_all = True
        for budget in BUDGETS:
            full = summarize_by_seed(rows, 'learned_full_aware', budget)
            bounded = summarize_by_seed(rows, f'learned_k{k}_aware', budget)
            gap = full - bounded
            gaps[str(budget)] = {'mean_full_minus_k': float(np.mean(gap)), 'ci95': list(bootstrap_seed_ci(gap, boot_rng))}
            if float(np.mean(gap)) > 0.01:
                ok_all = False
        capacity[str(k)] = gaps
        if minimal_k is None and ok_all:
            minimal_k = k
    pass_h1 = h1['mean'] >= 0.02 and h1['ci95'][0] > 0.0 and h1['positive_seeds'] >= 10 and h1_long['mean'] >= -0.01
    pass_latency = latency['mean'] > 0.0 and latency['ci95'][0] > 0.0 and latency_long['mean'] >= 0.0
    decision = 'PASS' if pass_h1 else 'FAIL'
    return {
        'name': 'AP-LM1 recursive deferred-query frontier',
        'construct': 'recursive LM-like querying with perfect answers, serial latency, and answer-generated unknowns',
        'fit_seed': FIT_SEED,
        'fit_episodes': FIT_EPISODES,
        'teacher_rows': teacher_rows,
        'test_seeds': list(TEST_SEEDS),
        'episodes_per_seed': EPISODES_PER_SEED,
        'budgets': list(BUDGETS),
        'k_values': list(K_VALUES),
        'aggregate': aggregate,
        'h1_full_aware_vs_greedy_b16': h1,
        'long_horizon_full_aware_vs_greedy_b32': h1_long,
        'latency_ablation_b16': latency,
        'latency_ablation_b32': latency_long,
        'capacity_full_minus_k': capacity,
        'minimal_k_within_1pp_of_full_at_both_budgets': minimal_k,
        'pass_h1': pass_h1,
        'pass_latency_ablation': pass_latency,
        'decision': decision,
        'boundaries': [
            'Hallucination is fixed to zero; this isolates recursive query selection and latency.',
            'Latency is serial query cost in AP-LM1; asynchronous overlap/slack is deferred to AP-LM2.',
            'Synthetic useful-information reward is not a human comprehension measure.',
            'Development pilot results are not included in confirmatory test seeds.',
        ],
    }

def write_outputs(results):
    outdir = Path('artifacts/ap_lm1')
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'AP_LM1_RESULTS.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    h1 = results['h1_full_aware_vs_greedy_b16']
    h32 = results['long_horizon_full_aware_vs_greedy_b32']
    lat = results['latency_ablation_b16']
    lines = [
        '# AP-LM1 Summary', '', f"Decision: **{results['decision']}**", '',
        '## Primary confirmatory comparison',
        f"- learned full latency-aware vs global visible greedy, B=16: {100 * h1['mean']:+.3f} pp oracle-normalized",
        f"- seed-cluster bootstrap 95% CI: [{100 * h1['ci95'][0]:+.3f}, {100 * h1['ci95'][1]:+.3f}] pp",
        f"- positive seeds: {h1['positive_seeds']}/12",
        f"- B=32 delta: {100 * h32['mean']:+.3f} pp", '',
        '## Latency ablation',
        f"- aware minus no-latency, B=16: {100 * lat['mean']:+.3f} pp, CI [{100 * lat['ci95'][0]:+.3f}, {100 * lat['ci95'][1]:+.3f}] pp", '',
        '## Bounded frontier capacity',
        f"- minimal K within 1 pp of full learned frontier at both budgets: {results['minimal_k_within_1pp_of_full_at_both_budgets']}", '',
        '## Interpretation boundary',
        'This is a controlled recursive-query model with perfect answers and serial latency. It tests query-frontier control, not hallucination or human comprehension.',
    ]
    (outdir / 'AP_LM1_SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

def main():
    model_aware, model_no_latency, teacher_rows = train_models()
    rows = run_confirmatory(model_aware, model_no_latency)
    results = make_results(rows, teacher_rows)
    write_outputs(results)
    print('AP_LM1_DECISION', results['decision'])
    print('AP_LM1_H1', json.dumps(results['h1_full_aware_vs_greedy_b16']))
    print('AP_LM1_LONG', json.dumps(results['long_horizon_full_aware_vs_greedy_b32']))
    print('AP_LM1_LATENCY', json.dumps(results['latency_ablation_b16']))
    print('AP_LM1_MIN_K', results['minimal_k_within_1pp_of_full_at_both_budgets'])

if __name__ == '__main__':
    main()
