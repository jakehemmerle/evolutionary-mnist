import hashlib
import math
import random
from dataclasses import asdict
from itertools import product

from config import HyperParams, TrainingHistory
from evolution.pareto import pareto_front


def _product_size(option_lists: dict[str, list]) -> int:
    n = 1
    for v in option_lists.values():
        n *= max(1, len(v))
    return n


def _stable_config_key(params: HyperParams) -> str:
    # Stable across processes/runs (unlike Python's hash()).
    blob = str(sorted(asdict(params).items())).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def enforce_cap_by_shrinking(option_lists: dict[str, list], cap: int) -> dict[str, list]:
    """
    Deterministically shrink option lists (drop from the end of the longest lists)
    until the Cartesian product size is <= cap.
    """
    if cap < 1:
        raise ValueError("cap must be >= 1")

    shrunk = {k: list(v) for k, v in option_lists.items()}
    while _product_size(shrunk) > cap:
        # Find the longest list with >1 option; break ties by key for determinism.
        candidates = [(len(v), k) for k, v in shrunk.items() if len(v) > 1]
        if not candidates:
            break
        _, key = max(candidates)
        shrunk[key] = shrunk[key][:-1]
    return shrunk


def expand_option_lists(
    option_lists: dict[str, list],
    cap: int,
    seed: int,
) -> list[HyperParams]:
    """
    Expand per-hparam option lists to explicit HyperParams via Cartesian product.
    Enforces cap by shrinking first, then (if needed) deterministic sampling.
    """
    option_lists = enforce_cap_by_shrinking(option_lists, cap=cap)

    keys = list(option_lists.keys())
    values = [option_lists[k] for k in keys]
    configs = [HyperParams(**dict(zip(keys, combo))) for combo in product(*values)]

    if len(configs) <= cap:
        return configs

    # Deterministic sampling fallback (should be rare given shrinking).
    rng = random.Random(seed)
    scored = []
    for p in configs:
        # deterministic per-config key, then a seeded shuffle over those keys
        scored.append((_stable_config_key(p), p))
    scored.sort(key=lambda t: t[0])
    rng.shuffle(scored)
    return [p for _, p in scored[:cap]]


def propose_next_generation_option_lists(
    base_search_space: dict[str, list],
    results_so_far: list[TrainingHistory],
    cap: int,
    seed: int,
) -> dict[str, list]:
    """
    Heuristic (non-LLM) proposal: anchor on best-so-far val_accuracy and vary up to log2(cap)
    hyperparameters by providing 2 options each (anchor + alternative), keeping others fixed.

    This produces a mini-grid whose size is <= cap.
    """
    if cap < 1:
        raise ValueError("cap must be >= 1")

    # Anchor: prefer the best-accuracy point on the Pareto front, else choose the first values.
    if results_so_far:
        front = pareto_front(results_so_far)
        anchor = max(front, key=lambda h: h.val_accuracy).params if front else max(results_so_far, key=lambda h: h.val_accuracy).params
    else:
        defaults = {k: v[0] for k, v in base_search_space.items() if v}
        anchor = HyperParams(**defaults)

    option_lists: dict[str, list] = {k: [getattr(anchor, k)] for k in base_search_space.keys()}

    # Vary a subset of params (2 options each) so product <= cap.
    max_binary_params = int(math.floor(math.log2(cap))) if cap > 1 else 0
    varyable = [k for k, v in base_search_space.items() if len(v) > 1]
    rng = random.Random(seed)
    rng.shuffle(varyable)
    varyable = varyable[:max_binary_params]

    for k in varyable:
        anchor_val = getattr(anchor, k)
        candidates = [x for x in base_search_space[k] if x != anchor_val]
        if not candidates:
            continue
        alt = candidates[0]
        option_lists[k] = [anchor_val, alt]

    # Final cap enforcement (handles weird non-binary list lengths).
    option_lists = enforce_cap_by_shrinking(option_lists, cap=cap)
    return option_lists


