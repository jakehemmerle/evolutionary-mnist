from __future__ import annotations

from dataclasses import asdict

from config import TrainingHistory


def dominates(a: TrainingHistory, b: TrainingHistory) -> bool:
    """
    Multi-objective dominance:
    - maximize val_accuracy
    - minimize wall_time_seconds
    - minimize param_count
    """
    a_acc, b_acc = a.val_accuracy, b.val_accuracy
    a_t, b_t = a.wall_time_seconds, b.wall_time_seconds
    a_p, b_p = a.param_count, b.param_count

    no_worse = (a_acc >= b_acc) and (a_t <= b_t) and (a_p <= b_p)
    strictly_better = (a_acc > b_acc) or (a_t < b_t) or (a_p < b_p)
    return no_worse and strictly_better


def non_dominated_sort(results: list[TrainingHistory]) -> list[list[TrainingHistory]]:
    """
    Returns Pareto fronts (front[0] is the non-dominated set).
    O(n^2) which is fine for small populations.
    """
    if not results:
        return []

    n = len(results)
    dominated_by_count = [0] * n
    dominates_sets: list[set[int]] = [set() for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(results[i], results[j]):
                dominates_sets[i].add(j)
            elif dominates(results[j], results[i]):
                dominated_by_count[i] += 1

    fronts: list[list[int]] = []
    current_front = [i for i in range(n) if dominated_by_count[i] == 0]

    remaining = set(range(n))
    while current_front:
        fronts.append(current_front)
        remaining -= set(current_front)
        next_front: list[int] = []
        for i in current_front:
            for j in dominates_sets[i]:
                dominated_by_count[j] -= 1
                if dominated_by_count[j] == 0:
                    next_front.append(j)
        current_front = [i for i in next_front if i in remaining]

    return [[results[i] for i in front] for front in fronts]


def pareto_front(results: list[TrainingHistory]) -> list[TrainingHistory]:
    fronts = non_dominated_sort(results)
    return fronts[0] if fronts else []


def select_elite(results: list[TrainingHistory], elite_k: int) -> list[TrainingHistory]:
    """
    Pick up to elite_k results preferring earlier Pareto fronts, then tie-breaking by:
    accuracy desc, time asc, params asc.
    """
    if elite_k <= 0 or not results:
        return []

    fronts = non_dominated_sort(results)
    elite: list[TrainingHistory] = []
    for front in fronts:
        front_sorted = sorted(
            front,
            key=lambda h: (-h.val_accuracy, h.wall_time_seconds, h.param_count, str(sorted(asdict(h.params).items()))),
        )
        for h in front_sorted:
            if len(elite) >= elite_k:
                return elite
            elite.append(h)
    return elite


