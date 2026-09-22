from cli import EpisodeSummary, aggregate_episode_summaries, format_stats_table


def test_aggregate_stats_and_priority_counts():
    stats = aggregate_episode_summaries([
        EpisodeSummary(2, 10, 12.5, (10, 20), 1, {"attack": 2}),
        EpisodeSummary(4, 14, 20.0, (30, 40), 0, {"attack": 1, "retreat": 3}),
    ])
    assert (stats.kills_min, stats.kills_avg, stats.kills_max) == (2, 3.0, 4)
    assert stats.survival_tics_avg == 12.0
    assert stats.total_reward == 32.5
    assert stats.latency_avg_ms == 25.0
    assert stats.latency_p95_ms == 38.5
    assert stats.rails_intervention_rate == 0.5
    assert stats.priority_fires == {"attack": 3, "retreat": 3}


def test_format_is_plain_text_and_includes_all_metrics():
    stats = aggregate_episode_summaries([EpisodeSummary(priority_fires={"defend": 1})])
    output = format_stats_table(stats)
    assert "kills (min/avg/max)" in output
    assert "latency ms (avg/p95)" in output
    assert "rails intervention rate" in output
    assert "priority fires: defend" in output
    assert "|" in output


def test_empty_latency_is_safe():
    stats = aggregate_episode_summaries([EpisodeSummary()])
    assert stats.latency_avg_ms == stats.latency_p95_ms == 0.0


def test_empty_summaries_are_rejected():
    try:
        aggregate_episode_summaries([])
    except ValueError as error:
        assert "at least one" in str(error)
    else:
        raise AssertionError("expected ValueError")
