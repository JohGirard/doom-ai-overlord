"""Command-line reporting for repeated episode runs.

The aggregation functions are intentionally independent of the game runner so
they can also be used by tests and by other frontends.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class EpisodeSummary:
    kills: int = 0
    survival_tics: int = 0
    reward: float = 0.0
    latency_ms: Sequence[float] = ()
    rails_interventions: int = 0
    priority_fires: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class AggregateStats:
    episodes: int
    kills_min: int
    kills_avg: float
    kills_max: int
    survival_tics_avg: float
    total_reward: float
    latency_avg_ms: float
    latency_p95_ms: float
    rails_intervention_rate: float
    priority_fires: Mapping[str, int]


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def aggregate_episode_summaries(summaries: Iterable[EpisodeSummary]) -> AggregateStats:
    items = list(summaries)
    if not items:
        raise ValueError("at least one episode summary is required")
    kills = [item.kills for item in items]
    latencies = [latency for item in items for latency in item.latency_ms]
    priorities: dict[str, int] = {}
    for item in items:
        for priority, count in item.priority_fires.items():
            priorities[priority] = priorities.get(priority, 0) + count
    return AggregateStats(
        episodes=len(items),
        kills_min=min(kills),
        kills_avg=sum(kills) / len(items),
        kills_max=max(kills),
        survival_tics_avg=sum(item.survival_tics for item in items) / len(items),
        total_reward=sum(item.reward for item in items),
        latency_avg_ms=sum(latencies) / len(latencies) if latencies else 0.0,
        latency_p95_ms=_percentile(latencies, 0.95),
        rails_intervention_rate=sum(item.rails_interventions for item in items) / len(items),
        priority_fires=dict(sorted(priorities.items())),
    )


def format_stats_table(stats: AggregateStats) -> str:
    rows = [
        ("episodes", str(stats.episodes)),
        ("kills (min/avg/max)", f"{stats.kills_min}/{stats.kills_avg:.2f}/{stats.kills_max}"),
        ("survival tics (avg)", f"{stats.survival_tics_avg:.2f}"),
        ("total reward", f"{stats.total_reward:.2f}"),
        ("latency ms (avg/p95)", f"{stats.latency_avg_ms:.2f}/{stats.latency_p95_ms:.2f}"),
        ("rails intervention rate", f"{stats.rails_intervention_rate:.2%}"),
    ]
    rows.extend((f"priority fires: {name}", str(count)) for name, count in stats.priority_fires.items())
    width = max(len(label) for label, _ in rows)
    return "Episode statistics\n" + "\n".join(f"{label:<{width}} | {value}" for label, value in rows)


def run_episode() -> EpisodeSummary:
    """Run one episode and return its summary.

    Projects embedding this standalone runner can replace this function with
    their game-specific episode execution without changing aggregation.
    """
    return EpisodeSummary()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate repeated episode statistics")
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    summaries = [run_episode() for _ in range(args.episodes)]
    print(format_stats_table(aggregate_episode_summaries(summaries)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
